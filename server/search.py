#!/usr/bin/env python3
"""
Unified Driving Scene Search Application

Features:
- Multi-dataset support
- Environment filters (weather, time_of_day, road_condition)
- Driving context filters (traffic density, pedestrian density)
- Object category filters
- Multi-modal search (text, image)
- CLIP embeddings (M-CLIP for text, CLIP for images)

Usage:
    pip install fastapi uvicorn neo4j clip-by-openai torch numpy pillow
    python search.py
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
from urllib.parse import quote

import torch
import clip
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from multilingual_clip import pt_multilingual_clip
import transformers

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from pydantic import BaseModel
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configure logging from environment variable
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    limit: int = 100
    page: int = 1  # Pagination: 1-based page number

    # Dataset filter
    datasets: List[str] = []  # Which datasets to search

    # Object filters
    object_filters: Dict[str, int] = {}  # {category: minimum_count}

    # Environment filters
    weather: List[str] = []  # ["clear", "rainy", "snowy", etc.]
    time_of_day: List[str] = []  # ["daytime", "night", "dawn/dusk"]
    road_condition: List[str] = []  # ["dry", "wet"]

    # Driving context filters
    traffic_density: List[str] = []  # ["none", "low", "medium", "high"]
    pedestrian_density: List[str] = []  # ["none", "low", "medium", "high"]

class ObjectComposition(BaseModel):
    composition: Dict[str, int]  # {category: count}
    total: int

class ScenarioTag(BaseModel):
    type: str
    timestamp: int
    timestep: float

class SearchResult(BaseModel):
    node_id: str
    dataset: str
    data_catalog_id: str
    token: str
    filepath: str
    original_filepath: str
    image_url: str
    score: float
    scores: Dict[str, float]
    objects: ObjectComposition
    environment: Dict[str, str]  # weather, time_of_day, road_condition
    context: Dict[str, str]  # traffic_density, pedestrian_density

class SearchScenarioResult(BaseModel):
    node_id: str
    dataset: str
    data_catalog_id: str
    scenario_id: str
    original_filepath: str
    scenario_tag: List[ScenarioTag]

class SearchScenarioResponse(BaseModel):
    results: List[SearchScenarioResult]
    page: int = 1
    has_more: bool = False

class SearchResponse(BaseModel):
    results: List[SearchResult]
    page: int = 1
    has_more: bool = False

class UnifiedSearchEngine:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
        self.total_sample_count = 0  # Cached total sample count
        # Device selection: environment variable overrides auto-detection
        device_env = os.getenv("DEVICE", "auto").lower()
        if device_env == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_env

        # CLIP for image encoder
        self.clip_model = None
        self.clip_preprocess = None

        # Multilingual CLIP for text queries (100+ languages)
        self.mclip_model = None
        self.mclip_tokenizer = None

        logger.info(f"Initialized search engine on device: {self.device}")

    def connect_neo4j(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection with optimized fetch size
            with self.driver.session(fetch_size=1000) as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def count_total_samples(self):
        """Count total samples with embeddings for pagination"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (s:Sample)
                    WHERE s._img_embedding_ViT_L_14 IS NOT NULL
                    RETURN count(s) as total
                """)
                record = result.single()
                self.total_sample_count = record['total'] if record else 0
                logger.info("Total samples with embeddings: %d", self.total_sample_count)
                return self.total_sample_count
        except Exception as e:
            logger.error("Failed to count samples: %s", e)
            return 0

    def ensure_vector_indexes(self):
        """Ensure vector indexes exist for efficient kNN search"""
        try:
            with self.driver.session() as session:
                # Get existing indexes
                result = session.run("SHOW INDEXES")
                index_names = {record['name'] for record in result}

                # Create CLIP image embedding index (768 dimensions)
                if 'clip_image_vector_index' not in index_names:
                    logger.info("Creating CLIP image vector index...")
                    session.run("""
                    CREATE VECTOR INDEX clip_image_vector_index IF NOT EXISTS
                    FOR (s:Sample)
                    ON s._img_embedding_ViT_L_14
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                    """)
                    logger.info("Created CLIP image vector index")

                # Create Eval-specific CLIP image embedding index
                if 'eval_clip_image_vector_index' not in index_names:
                    logger.info("Creating Eval-specific CLIP image vector index...")
                    session.run("""
                    CREATE VECTOR INDEX eval_clip_image_vector_index IF NOT EXISTS
                    FOR (s:Eval)
                    ON s._img_embedding_ViT_L_14
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                    """)
                    logger.info("Created Eval-specific CLIP image vector index")

            logger.info("Vector indexes are ready")
            return True

        except Exception as e:
            logger.error(f"Failed to ensure vector indexes: {e}")
            return False

    def load_models(self):
        """Load Multilingual CLIP models"""
        try:
            logger.info("=" * 60)
            logger.info("Loading models on device: %s", self.device)
            logger.info("=" * 60)

            # Load Multilingual CLIP model (for Korean/English/100+ languages text)
            logger.info("[1/2] Loading M-CLIP (XLM-Roberta-Large-Vit-L-14) for multilingual text...")
            self.mclip_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
            self.mclip_tokenizer = transformers.AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
            logger.info("   M-CLIP loaded successfully (supports 100+ languages)")

            # Load CLIP and preprocessing (for images)
            logger.info("[2/2] Loading CLIP (ViT-L/14) visual model for images...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
            self.clip_model.eval()
            logger.info("   CLIP loaded successfully (768 dimensions)")

            logger.info("=" * 60)
            logger.info("All models loaded successfully!")
            logger.info("=" * 60)
            return True
        except Exception as e:
            logger.error("Failed to load models: %s", e)
            return False

    def generate_query_embeddings(self, query: str) -> Dict[str, Optional[List[float]]]:
        """Generate embeddings for query text using Multilingual CLIP (supports Korean, English, 100+ languages)"""
        embeddings = {}

        try:
            # Multilingual CLIP text embedding (Korean/English/100+ languages)
            # This uses M-CLIP which is compatible with OpenAI CLIP image embeddings
            text_emb = self.mclip_model.forward([query], self.mclip_tokenizer)
            embeddings['clip_text'] = text_emb.detach().cpu().numpy()[0].flatten().tolist()

            # Add original query text for keyword matching
            embeddings['query_text'] = query

            query_preview = query[:50] + '...' if len(query) > 50 else query
            logger.info("Generated text embeddings for query: '%s'", query_preview)
            return embeddings

        except Exception as e:
            logger.error("Failed to generate query embeddings: %s", e)
            return {'clip_text': None, 'query_text': ''}

    def generate_image_embeddings(self, image_bytes: bytes) -> Dict[str, Optional[List[float]]]:
        """Generate embeddings for uploaded image using CLIP"""
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Preprocess image using CLIP transforms
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            # Generate image embedding using CLIP
            with torch.no_grad():
                clip_image_embedding = self.clip_model.encode_image(image_tensor)
                clip_image_embedding = clip_image_embedding.detach().cpu().numpy().flatten().tolist()

            logger.info("Generated image embedding: %d dimensions, size: %s",
                       len(clip_image_embedding), image.size)
            return {
                'clip_image': clip_image_embedding,
                'image_size': image.size,
                'image_mode': image.mode
            }

        except Exception as e:
            logger.error("Failed to generate image embeddings: %s", e)
            return {'clip_image': None, 'image_size': None, 'image_mode': None}

    def build_search_query_knn(self, datasets: List[str], object_filters: Dict[str, int],
                          weather: List[str], time_of_day: List[str], road_condition: List[str],
                          traffic_density: List[str], pedestrian_density: List[str],
                          limit: int, page: int = 1,
                          query_embeddings: Dict[str, List[float]] = None,
                          image_embeddings: Dict[str, List[float]] = None,
                          image_weight: float = 0.6, text_weight: float = 0.4,
                          mode: str = None) -> tuple:
        # Get embeddings
        clip_image_query = image_embeddings.get('clip_image') if image_embeddings else None
        clip_text_query = query_embeddings.get('clip_text') if query_embeddings else None

        # Determine which vector index to use based on mode
        # CLIP's image and text embeddings share the same latent space
        if mode == 'eval':
            # Use Eval-specific vector index for evaluation
            vector_index = 'eval_clip_image_vector_index'
        else:
            # Use general vector index
            vector_index = 'clip_image_vector_index'

        if clip_image_query:
            primary_vector = clip_image_query
            primary_index = vector_index
        elif clip_text_query:
            primary_vector = clip_text_query
            primary_index = vector_index
        else:
            return self._build_filter_only_query(datasets, object_filters, weather,
                                                 time_of_day, road_condition,
                                                 traffic_density, pedestrian_density, limit, page)

        # Fixed k_candidates with cap for performance
        k_candidates = min(page * limit * 50, 50000)  # Cap at 50k
        skip_count = (page - 1) * limit
        fetch_limit = limit + 1  # Fetch one extra to determine has_more

        logger.info("kNN search: page=%d, limit=%d, skip=%d, k_candidates=%d",
                   page, limit, skip_count, k_candidates)

        # Build kNN query with post-filtering
        query = f"""
        CALL db.index.vector.queryNodes($primary_index, $k_candidates, $primary_vector)
        YIELD node as s, score as knn_score
        """

        # Add filter conditions
        filter_conditions = []

        # Dataset filter (skip for eval mode by not passing datasets parameter)
        if datasets:
            filter_conditions.append("ds.name IN $datasets")

        # Check required properties exist
        filter_conditions.append("s._img_embedding_ViT_L_14 IS NOT NULL")
        filter_conditions.append("s.filepath IS NOT NULL")

        # Add relationships and environment filters
        query += """
        MATCH (s)-[:IN_CATALOG]->(ds:DataCatalog)
        MATCH (s)-[:HAS_WEATHER]->(w:Weather)
        MATCH (s)-[:HAS_TIME_OF_DAY]->(t:TimeOfDay)
        MATCH (s)-[:HAS_ROAD_CONDITION]->(r:RoadCondition)
        MATCH (s)-[:HAS_CONTEXT]->(c:DrivingContext)
        """

        # Add environment filter conditions
        if weather:
            filter_conditions.append("w.name IN $weather")
        if time_of_day:
            filter_conditions.append("t.name IN $time_of_day")
        if road_condition:
            filter_conditions.append("r.name IN $road_condition")
        if traffic_density:
            filter_conditions.append("c.traffic_density IN $traffic_density")
        if pedestrian_density:
            filter_conditions.append("c.pedestrian_density IN $pedestrian_density")

        # Add object filters
        if object_filters:
            query += """
            OPTIONAL MATCH (s)-[det_rel:DETECTED_OBJECT]->(o:Object)
            WITH s, ds, w, t, r, c, knn_score, o, det_rel
            """

            # Calculate category counts
            category_counts = []
            for i, (category, min_count) in enumerate(object_filters.items()):
                category_var = category.lower().replace(' ', '_').replace('/', '_')
                category_counts.append(f"""
                    sum(CASE WHEN o.label = $category_{i} THEN det_rel.count ELSE 0 END) as {category_var}_count
                """)

            query += f"""
            WITH s, ds, w, t, r, c, knn_score,
                {','.join(category_counts)}
            """

            # Add object count filter conditions
            for category, min_count in object_filters.items():
                category_var = category.lower().replace(' ', '_').replace('/', '_')
                filter_conditions.append(f"{category_var}_count >= $min_{category_var}")

        # Apply all filter conditions
        if filter_conditions:
            query += f"WHERE {' AND '.join(filter_conditions)}\n"

        # Prepare category variables for WITH clause if object filters exist
        category_vars_str = ''
        if object_filters:
            category_vars = ', '.join([f"{cat.lower().replace(' ', '_').replace('/', '_')}_count"
                                      for cat in object_filters.keys()])
            category_vars_str = f"{category_vars}, "

        # Calculate similarity scores
        if clip_image_query and clip_text_query:
            # HYBRID MODE: Both image and text
            # Calculate image similarity from kNN (primary search with image)
            # Calculate text similarity by comparing text query with image embeddings (same CLIP space)
            query += f"""
            WITH s, ds, w, t, r, c, knn_score, {category_vars_str}
                knn_score as img_similarity,
                vector.similarity.cosine(s._img_embedding_ViT_L_14, $clip_text_query) as txt_similarity
            WITH s, ds, w, t, r, c, {category_vars_str}img_similarity, txt_similarity,
                ($image_weight * img_similarity + $text_weight * txt_similarity) as final_score
            """
        elif clip_image_query:
            # IMAGE-ONLY MODE
            query += f"""
            WITH s, ds, w, t, r, c, {category_vars_str}knn_score as final_score,
            knn_score as img_similarity,
            0.0 as txt_similarity
            """
        elif clip_text_query:
            # TEXT-ONLY MODE
            query += f"""
            WITH s, ds, w, t, r, c, {category_vars_str}knn_score as final_score,
            0.0 as img_similarity,
            knn_score as txt_similarity
            """

        # Build RETURN clause
        return_fields = [
            "elementId(s) as node_id",
            "ds.source as dataset_source",
            "ds.name as dataset",
            "ds.catalog_id as data_catalog_id",
            "s.filepath as filepath",
            "s._key as token",
            "img_similarity",
            "txt_similarity",
            "final_score",
            "w.name as weather",
            "t.name as time_of_day",
            "r.name as road_condition",
            "c.traffic_density as traffic_density",
            "c.pedestrian_density as pedestrian_density"
        ]

        # Add object count fields
        if object_filters:
            for category in object_filters.keys():
                category_var = category.lower().replace(' ', '_').replace('/', '_')
                return_fields.append(f"{category_var}_count")

        # Add RETURN, ORDER BY, SKIP, and LIMIT
        query += f"""
                RETURN {', '.join(return_fields)}
                ORDER BY final_score DESC
                SKIP $skip_count
                LIMIT $fetch_limit
                """

        # Prepare parameters
        query_params = {
            'primary_index': primary_index,
            'k_candidates': k_candidates,
            'primary_vector': primary_vector,
            'skip_count': skip_count,
            'fetch_limit': fetch_limit,
            'datasets': datasets
        }

        # Add embeddings for hybrid mode similarity calculation
        if clip_image_query and clip_text_query:
            query_params['clip_text_query'] = clip_text_query
            query_params['image_weight'] = image_weight
            query_params['text_weight'] = text_weight

        # Add filter parameters
        if weather:
            query_params['weather'] = weather
        if time_of_day:
            query_params['time_of_day'] = time_of_day
        if road_condition:
            query_params['road_condition'] = road_condition
        if traffic_density:
            query_params['traffic_density'] = traffic_density
        if pedestrian_density:
            query_params['pedestrian_density'] = pedestrian_density

        # Add object filter parameters
        if object_filters:
            for i, (category, min_count) in enumerate(object_filters.items()):
                query_params[f'category_{i}'] = category
                query_params[f'min_{category.lower().replace(" ", "_").replace("/", "_")}'] = min_count

        return query, query_params

    def _build_scenario_filter_query(self, object_filters: Dict[str, int],
                                 scenario_tag: List[str], limit: int, page: int = 1) -> tuple:

        query = f"""
        MATCH (s:Scene)-[:HAS_EVENT]->(se:ScenarioEvent)
        """

        filter_conditions = []

        if scenario_tag:
            query += "WHERE se.type IN $scenario_tag"

        query += f"""
        WITH s, collect(DISTINCT se) as scenario_events
        WITH s, [se IN scenario_events | {{
            type: se.type,
            timestamp: se.timestamp,
            timestep: se.timestep
        }}] as scenario_tag
        MATCH (s)-[:IN_CATALOG]->(ds:DataCatalog)
        """


        if filter_conditions:
            query += f"WHERE {' AND '.join(filter_conditions)}"

        # Return fields
        return_fields = [
            "elementId(s) as node_id",
            "ds.name as dataset",
            "ds.catalog_id as data_catalog_id",
            "s.scenario_id as scenario_id",
            "scenario_tag"        
        ]

        skip_count = (page - 1) * limit
        fetch_limit = limit + 1

        query += f"""
        RETURN {', '.join(return_fields)}
        ORDER BY id(s)
        SKIP $skip_count
        LIMIT $fetch_limit
        """

        query_params = {'skip_count': skip_count, 'fetch_limit': fetch_limit}

        if scenario_tag:
            query_params['scenario_tag'] = scenario_tag
        
        return query, query_params

    def _build_filter_only_query(self, datasets: List[str], object_filters: Dict[str, int],
                                 weather: List[str], time_of_day: List[str],
                                 road_condition: List[str], traffic_density: List[str],
                                 pedestrian_density: List[str], limit: int, page: int = 1) -> tuple:
        """Build filter-only query when no embeddings are provided"""

        # Build WHERE clause
        where_conditions = ["s.filepath IS NOT NULL"]
        if datasets:
            where_conditions.insert(0, "ds.name IN $datasets")

        query = f"""
        MATCH (s:Sample)-[:IN_CATALOG]->(ds:DataCatalog)
        WHERE {' AND '.join(where_conditions)}
        """

        filter_conditions = []
        return_fields = [
            "elementId(s) as node_id",
            "ds.source as dataset_source",
            "ds.name as dataset",
            "ds.catalog_id as data_catalog_id",
            "s.filepath as filepath",
            "s._key as token",
            "w.name as weather",
            "t.name as time_of_day",
            "r.name as road_condition",
            "c.traffic_density as traffic_density",
            "c.pedestrian_density as pedestrian_density",
        ]

        # 필터 사용 시에만 MATCH + WHERE로 조기 필터링 (성능). 변수는 wf,tf,rf,cf로 구분해 마지막 OPTIONAL MATCH(w,t,r,c)와 분리
        if weather:
            query += "\n        MATCH (s)-[:HAS_WEATHER]->(wf:Weather) WHERE wf.name IN $weather"
        if time_of_day:
            query += "\n        MATCH (s)-[:HAS_TIME_OF_DAY]->(tf:TimeOfDay) WHERE tf.name IN $time_of_day"
        if road_condition:
            query += "\n        MATCH (s)-[:HAS_ROAD_CONDITION]->(rf:RoadCondition) WHERE rf.name IN $road_condition"
        if traffic_density or pedestrian_density:
            c_where = []
            if traffic_density:
                c_where.append("cf.traffic_density IN $traffic_density")
            if pedestrian_density:
                c_where.append("cf.pedestrian_density IN $pedestrian_density")
            query += "\n        MATCH (s)-[:HAS_CONTEXT]->(cf:DrivingContext) WHERE " + " AND ".join(c_where)

        if object_filters:
            category_counts = []
            for i, (category, min_count) in enumerate(object_filters.items()):
                category_var = category.lower().replace(' ', '_').replace('/', '_')
                category_counts.append(f"""
                    sum(CASE WHEN o.label = $category_{i} THEN det_rel.count ELSE 0 END) as {category_var}_count
                """)

            query += f"""
            MATCH (s:Sample)-[det_rel:DETECTED_OBJECT]->(o:Object)
            WITH s, ds,
                {','.join(category_counts)}
            """

            for category, min_count in object_filters.items():
                category_var = category.lower().replace(' ', '_').replace('/', '_')
                filter_conditions.append(f"{category_var}_count >= $min_{category_var}")

        if filter_conditions:
            query += f"WHERE {' AND '.join(filter_conditions)}\n"

        if object_filters:
            for category in object_filters.keys():
                category_var = category.lower().replace(' ', '_').replace('/', '_')
                return_fields.append(f"{category_var}_count")

        # SKIP/LIMIT을 먼저 적용해 페이지에 필요한 (s)만 남긴 뒤, OPTIONAL MATCH로 표시용 속성 조회
        skip_count = (page - 1) * limit
        fetch_limit = limit + 1
        pagination_with = ["s", "ds"]
        if object_filters:
            pagination_with += [f"{c.lower().replace(' ', '_').replace('/', '_')}_count" for c in object_filters.keys()]

        query += f"""
        WITH {', '.join(pagination_with)}
        ORDER BY id(s)
        SKIP $skip_count
        LIMIT $fetch_limit
        OPTIONAL MATCH (s)-[:HAS_WEATHER]->(w:Weather)
        OPTIONAL MATCH (s)-[:HAS_TIME_OF_DAY]->(t:TimeOfDay)
        OPTIONAL MATCH (s)-[:HAS_ROAD_CONDITION]->(r:RoadCondition)
        OPTIONAL MATCH (s)-[:HAS_CONTEXT]->(c:DrivingContext)
        RETURN {', '.join(return_fields)}
        """

        query_params = {'skip_count': skip_count, 'fetch_limit': fetch_limit, 'datasets': datasets}

        if weather:
            query_params['weather'] = weather
        if time_of_day:
            query_params['time_of_day'] = time_of_day
        if road_condition:
            query_params['road_condition'] = road_condition
        if traffic_density:
            query_params['traffic_density'] = traffic_density
        if pedestrian_density:
            query_params['pedestrian_density'] = pedestrian_density

        if object_filters:
            for i, (category, min_count) in enumerate(object_filters.items()):
                query_params[f'category_{i}'] = category
                query_params[f'min_{category.lower().replace(" ", "_").replace("/", "_")}'] = min_count

        #logger.info("Query: %s", query)

        return query, query_params

    def search_scenario_nodes(self, scenario_tag: List[str],
                     object_filters: Dict[str, int] = None,
                     limit: int = 24,
                     page: int = 1) -> Dict:

        # PERFORMANCE: Start timing
        t_start = time.time()

        # Set defaults for filter parameters
        object_filters = object_filters or {}
        scenario_tag = scenario_tag or []

        # Determine search type for logging
        search_type = "SCENARIO-FILTER"

        # Use kNN search query with vector indexes
        t_query_build_start = time.time()
        query, query_params = self._build_scenario_filter_query(
            object_filters, scenario_tag, limit, page
        )
        t_query_build = time.time() - t_query_build_start
        logger.info("[%s] Query build: %.2fms", search_type, t_query_build * 1000)
        try:
            with self.driver.session(fetch_size=1000) as session:
                t_db_start = time.time()
                result = session.run(query, **query_params)

                nodes = []
                record_count = 0
                
                for record in result:
                    record_count += 1
                    
                    # Extract required fields (always present)
                    node_id = record['node_id']
                    dataset = record['dataset']
                    data_catalog_id = record['data_catalog_id']
                    scenario_id = record['scenario_id']
                    scenario_tag = record['scenario_tag']
                    
                    nodes.append({
                        'node_id': node_id,
                        'dataset': dataset,
                        'data_catalog_id': data_catalog_id,
                        'scenario_id': scenario_id,
                        'scenario_tag': scenario_tag,
                    })

                t_db_total = time.time() - t_db_start
                t_total = time.time() - t_start

                logger.info("[%s] Neo4j query + fetch: %.2fms", search_type, t_db_total * 1000)
                logger.info("[%s] Total search time: %.2fms (found %d results, page %d)",
                           search_type, t_total * 1000, len(nodes), page)

                return {
                    'nodes': nodes[:limit],
                    'has_more': False
                }

        except Exception as e:
            logger.error("Search failed: %s", e)
            logger.error("Query params: %s", list(query_params.keys()))
            return {'nodes': [], 'has_more': False}

    def search_nodes(self, query_embeddings: Dict[str, List[float]] = None,
                     image_embeddings: Dict[str, List[float]] = None,
                     image_weight: float = 0.6, text_weight: float = 0.4,
                     mode: str = None,
                     datasets: List[str] = None,
                     object_filters: Dict[str, int] = None,
                     weather: List[str] = None,
                     time_of_day: List[str] = None,
                     road_condition: List[str] = None,
                     traffic_density: List[str] = None,
                     pedestrian_density: List[str] = None,
                     limit: int = 24,
                     page: int = 1) -> Dict:
        """Unified search supporting text-only, image-only, and hybrid (image+text) search with pagination

        Returns:
            Dict with 'nodes', 'has_more' keys
        """

        # PERFORMANCE: Start timing
        t_start = time.time()

        # Set defaults for filter parameters
        datasets = datasets or []
        object_filters = object_filters or {}
        weather = weather or []
        time_of_day = time_of_day or []
        road_condition = road_condition or []
        traffic_density = traffic_density or []
        pedestrian_density = pedestrian_density or []

        # Determine search type for logging
        search_type = "TEXT"
        if image_embeddings and query_embeddings:
            search_type = "HYBRID"
        elif image_embeddings:
            search_type = "IMAGE"
        elif not query_embeddings:
            search_type = "FILTER-ONLY"

        # Use kNN search query with vector indexes
        t_query_build_start = time.time()
        query, query_params = self.build_search_query_knn(
            datasets, object_filters, weather, time_of_day, road_condition,
            traffic_density, pedestrian_density, limit, page,
            query_embeddings=query_embeddings,
            image_embeddings=image_embeddings,
            image_weight=image_weight, text_weight=text_weight,
            mode=mode
        )
        t_query_build = time.time() - t_query_build_start
        logger.info("[%s] Query build: %.2fms", search_type, t_query_build * 1000)
        #logger.info("Query: %s", query)
        try:
            with self.driver.session(fetch_size=1000) as session:
                t_db_start = time.time()
                result = session.run(query, **query_params)

                nodes = []
                record_count = 0
                
                for record in result:
                    record_count += 1
                    
                    # Extract required fields (always present)
                    node_id = record['node_id']
                    data_catalog_id = record['data_catalog_id']
                    token = record['token']
                    filepath = record['filepath']
                    final_score = record.get('final_score', 0.0)
                    
                    # Extract similarity scores
                    img_similarity = record.get('img_similarity', 0.0)
                    txt_similarity = record.get('txt_similarity', 0.0)
                    
                    # Extract optional dataset fields
                    dataset_source = record.get('dataset_source', '')
                    dataset = record.get('dataset', '')
                    
                    # Extract optional environment fields
                    weather = record.get('weather') or 'undefined'
                    time_of_day = record.get('time_of_day') or 'undefined'
                    road_condition = record.get('road_condition') or 'undefined'
                    
                    # Extract optional context fields
                    traffic_density = record.get('traffic_density') or 'none'
                    pedestrian_density = record.get('pedestrian_density') or 'none'
                    
                    # Build object composition
                    object_composition = {}
                    total_objects = 0
                    if object_filters:
                        for category in object_filters.keys():
                            count_var = f"{category.lower().replace(' ', '_').replace('/', '_')}_count"
                            count = record.get(count_var, 0) or 0
                            object_composition[category] = count
                            total_objects += count

                    nodes.append({
                        'node_id': node_id,
                        'dataset': dataset,
                        'dataset_source': dataset_source,
                        'data_catalog_id': data_catalog_id,
                        'token': token,
                        'filepath': filepath,
                        'score': final_score,
                        'scores': {
                            'image': img_similarity,
                            'text': txt_similarity
                        },
                        'objects': {
                            'composition': object_composition,
                            'total': total_objects
                        },
                        'environment': {
                            'weather': weather,
                            'time_of_day': time_of_day,
                            'road_condition': road_condition
                        },
                        'context': {
                            'traffic_density': traffic_density,
                            'pedestrian_density': pedestrian_density
                        }
                    })

                t_db_total = time.time() - t_db_start
                t_total = time.time() - t_start

                logger.info("[%s] Neo4j query + fetch: %.2fms", search_type, t_db_total * 1000)
                logger.info("[%s] Total search time: %.2fms (found %d results, page %d)",
                           search_type, t_total * 1000, len(nodes), page)

                # Check if there are more results
                # has_more is true if we haven't reached the k_candidates cap
                k_candidates_used = min(page * limit * 50, 50000)
                has_more = k_candidates_used < 50000 or len(nodes) > limit

                logger.info("Pagination: page=%d, k_candidates=%d/50000, has_more=%s",
                           page, k_candidates_used, has_more)

                return {
                    'nodes': nodes[:limit],
                    'has_more': has_more
                }

        except Exception as e:
            logger.error("Search failed: %s", e)
            logger.error("Query params: %s", list(query_params.keys()))
            return {'nodes': [], 'has_more': False}

    def get_available_categories(self) -> List[str]:
        """Get all available object categories"""
        query = """
        MATCH (o:Object)
        WHERE NOT o.label = 'Undefined'
        RETURN o.label as category
        ORDER BY category
        """

        try:
            with self.driver.session(fetch_size=1000) as session:
                result = session.run(query)
                categories = []
                for record in result:
                    if record['category']:
                        categories.append(record['category'])
                return categories
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []

    def get_filter_options(self) -> Dict[str, List[str]]:
        """Get available filter options from graph"""
        try:
            with self.driver.session(fetch_size=1000) as session:
                # Get weather options
                weather_result = session.run("MATCH (w:Weather) WHERE NOT w.name = 'undefined' RETURN DISTINCT w.name as name")
                weather = [r['name'] for r in weather_result if r['name']]

                # Get time of day options
                time_result = session.run("MATCH (t:TimeOfDay) WHERE NOT t.name = 'undefined' RETURN t.name as name")
                time_of_day = [r['name'] for r in time_result if r['name']]

                # Get road condition options
                road_result = session.run("MATCH (r:RoadCondition) WHERE NOT r.name = 'undefined' RETURN r.name as name")
                road_condition = [r['name'] for r in road_result if r['name']]

                # Get traffic density options
                traffic_result = session.run("MATCH (c:DrivingContext) RETURN DISTINCT c.traffic_density as name")
                traffic_density = [r['name'] for r in traffic_result if r['name']]

                # Get pedestrian density options
                pedestrian_result = session.run("MATCH (c:DrivingContext) RETURN DISTINCT c.pedestrian_density as name")
                pedestrian_density = [r['name'] for r in pedestrian_result if r['name']]

                # Get dataset options   
                dataset_result = session.run("MATCH (n:DataCatalog) WHERE NOT n.name = 'Eval' RETURN n.name as name")
                datasets = [r['name'] for r in dataset_result if r['name']]

                # Get scenario tags   
                scenario_tag_result = session.run("MATCH (e:ScenarioEvent) RETURN e.category AS category,collect(DISTINCT e.type) AS types ORDER BY category")
                scenario_tags = {r['category']: sorted(r['types']) for r in scenario_tag_result if r['category'] and r['types']}

                return {
                    'datasets': sorted(datasets),
                    'weather': sorted(weather),
                    'time_of_day': sorted(time_of_day),
                    'road_condition': sorted(road_condition),
                    'traffic_density': sorted(traffic_density),
                    'pedestrian_density': sorted(pedestrian_density),
                    'scenario_tags': scenario_tags
                }
        except Exception as e:
            logger.error(f"Failed to get filter options: {e}")
            return {
                'datasets': [],
                'weather': [],
                'time_of_day': [],
                'road_condition': [],
                'traffic_density': [],
                'pedestrian_density': [],
                'scenario_tags': {}
            }

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

# Initialize search engine from environment variables
search_engine = UnifiedSearchEngine(
    neo4j_uri=os.getenv("NEO4J_URL", "bolt://54.249.172.114:7687"),
    neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
    neo4j_password=os.getenv("NEO4J_PASSWORD", "bg.cha@Int2.us")
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    # Startup
    if not search_engine.connect_neo4j():
        raise Exception("Failed to connect to Neo4j")

    if not search_engine.ensure_vector_indexes():
        raise Exception("Failed to ensure vector indexes")

    # Cache total sample count for pagination
    search_engine.count_total_samples()

    if not search_engine.load_models():
        raise Exception("Failed to load models")

    yield

    # Shutdown
    search_engine.close()

# FastAPI app configuration from environment
app = FastAPI(
    title=os.getenv("API_TITLE", "Unified Driving Scene Search"),
    version=os.getenv("API_VERSION", "2.0.0"),
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main search interface"""
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/scenario", response_model=SearchScenarioResponse)
async def search_scenario(
    object_filters: str = Form("{}"),
    scenario_tag: str = Form("[]"),
    limit: int = Form(100),
    page: int = Form(1)
):
    try:
        # PERFORMANCE: Start timing for entire API request
        t_api_start = time.time()

        # Parse filter parameters
        try:
            scenario_tag_list = json.loads(scenario_tag) if scenario_tag else []
            object_filters_dict = json.loads(object_filters) if object_filters else {}
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in parameters: {str(e)}")

        search_type = "SCENARIO-FILTER"

        logger.info("[%s] Search started: limit=%d, page=%d", search_type, limit, page)

        # Search with unified function
        t_search_start = time.time()
        search_result = search_engine.search_scenario_nodes(
            scenario_tag=scenario_tag_list,
            object_filters=object_filters_dict,
            limit=limit,
            page=page
        )
        t_search = time.time() - t_search_start

        nodes = search_result['nodes']
        has_more = search_result['has_more']

        # Format results
        t_format_start = time.time()
        results = []
        for node in nodes:
            results.append(SearchScenarioResult(
                node_id=node['node_id'],
                dataset=node['dataset'],
                data_catalog_id=node['data_catalog_id'],
                scenario_id=node['scenario_id'],
                original_filepath= "https://bdd100k-dataset-images.s3.ap-northeast-1.amazonaws.com/train/0000f77c-6257be58.jpg",
                scenario_tag=node['scenario_tag']
            ))

        t_format = time.time() - t_format_start
        t_api_total = time.time() - t_api_start

        logger.info("[%s API] Completed in %.2fms: %d results, page %d, has_more=%s (search: %.1fms, format: %.1fms)",
                   search_type, t_api_total * 1000, len(results), page, has_more,
                   t_search * 1000, t_format * 1000)

        return SearchScenarioResponse(results=results, page=page, has_more=has_more)

    except Exception as e:
        logger.error("[API] Search request failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_unified(
    query: str = Form(""),
    image: Optional[UploadFile] = File(None),
    image_weight: float = Form(0.6),
    text_weight: float = Form(0.4),
    mode: str = Form(None),
    datasets: str = Form('[]'),
    object_filters: str = Form("{}"),
    weather: str = Form("[]"),
    time_of_day: str = Form("[]"),
    road_condition: str = Form("[]"),
    traffic_density: str = Form("[]"),
    pedestrian_density: str = Form("[]"),
    limit: int = Form(100),
    page: int = Form(1)
):
    """Unified search endpoint supporting text-only, image-only, and hybrid (image+text) search"""
    try:
        # PERFORMANCE: Start timing for entire API request
        t_api_start = time.time()

        # Parse filter parameters
        try:
            datasets_list = json.loads(datasets) if datasets else []
            object_filters_dict = json.loads(object_filters) if object_filters else {}
            weather_list = json.loads(weather) if weather else []
            time_of_day_list = json.loads(time_of_day) if time_of_day else []
            road_condition_list = json.loads(road_condition) if road_condition else []
            traffic_density_list = json.loads(traffic_density) if traffic_density else []
            pedestrian_density_list = json.loads(pedestrian_density) if pedestrian_density else []
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in parameters: {str(e)}")

        # Generate text embeddings if query provided
        query_embeddings = None
        query_text = query.strip() if query else ""
        if query_text:
            t_embed_start = time.time()
            query_embeddings = search_engine.generate_query_embeddings(query_text)
            if not query_embeddings.get('clip_text'):
                raise HTTPException(status_code=500, detail="Failed to generate text embeddings")
            t_embed = time.time() - t_embed_start
            logger.info("[API] Text embedding: %.2fms", t_embed * 1000)

        # Generate image embeddings if image provided
        image_embeddings = None
        if image:
            # Validate image file
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")

            if image.size and image.size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")

            # Read image bytes
            image_bytes = await image.read()
            logger.info("Uploaded image: %s (%.2f KB)", image.filename, len(image_bytes) / 1024)

            t_img_embed_start = time.time()
            image_embeddings = search_engine.generate_image_embeddings(image_bytes)
            if not image_embeddings.get('clip_image'):
                raise HTTPException(status_code=500, detail="Failed to generate image embeddings")
            t_img_embed = time.time() - t_img_embed_start
            logger.info("[API] Image embedding: %.2fms", t_img_embed * 1000)

        # Determine search type
        if image_embeddings and query_embeddings:
            search_type = "HYBRID"
        elif image_embeddings:
            search_type = "IMAGE"
        elif query_embeddings:
            search_type = "TEXT"
        else:
            search_type = "FILTER-ONLY"

        logger.info("[%s] Search started: datasets=%s, limit=%d, page=%d",
                   search_type, datasets_list, limit, page)

        # Search with unified function
        t_search_start = time.time()
        search_result = search_engine.search_nodes(
            query_embeddings=query_embeddings,
            image_embeddings=image_embeddings,
            image_weight=image_weight,
            text_weight=text_weight,
            mode=mode,
            datasets=datasets_list,
            object_filters=object_filters_dict,
            weather=weather_list,
            time_of_day=time_of_day_list,
            road_condition=road_condition_list,
            traffic_density=traffic_density_list,
            pedestrian_density=pedestrian_density_list,
            limit=limit,
            page=page
        )
        t_search = time.time() - t_search_start

        nodes = search_result['nodes']
        has_more = search_result['has_more']

        # Format results
        t_format_start = time.time()
        results = []
        for node in nodes:
            # URL encode the filepath to handle special characters like +
            encoded_filepath = quote(node['filepath'], safe='/')
            results.append(SearchResult(
                node_id=node['node_id'],
                dataset=node['dataset'],
                data_catalog_id=node['data_catalog_id'],
                token=node['token'],
                filepath=node['filepath'],
                original_filepath= "https://bdd100k-dataset-images.s3.ap-northeast-1.amazonaws.com/train/0000f77c-6257be58.jpg",
                image_url=f"{node['dataset_source']}/{encoded_filepath}",
                score=node['score'],
                scores=node['scores'],
                objects=ObjectComposition(
                    composition=node['objects']['composition'],
                    total=node['objects']['total']
                ),
                environment=node['environment'],
                context=node['context']
            ))

        t_format = time.time() - t_format_start
        t_api_total = time.time() - t_api_start

        logger.info("[%s API] Completed in %.2fms: %d results, page %d, has_more=%s (search: %.1fms, format: %.1fms)",
                   search_type, t_api_total * 1000, len(results), page, has_more,
                   t_search * 1000, t_format * 1000)

        return SearchResponse(results=results, page=page, has_more=has_more)

    except Exception as e:
        logger.error("[API] Search request failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search/categories")
async def get_categories():
    """Get available object categories"""
    try:
        categories = search_engine.get_available_categories()
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@app.get("/filters/options")
async def get_filter_options():
    """Get available filter options"""
    try:
        options = search_engine.get_filter_options()
        return options
    except Exception as e:
        logger.error(f"Failed to get filter options: {e}")
        # Return empty options with error message to avoid breaking UI
        return {
            'datasets': [],
            'weather': [],
            'time_of_day': [],
            'road_condition': [],
            'traffic_density': [],
            'pedestrian_density': [],
            'error': f"Failed to get filter options: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if models are actually loaded
    models_loaded = (
        search_engine.clip_model is not None and
        search_engine.clip_preprocess is not None and
        search_engine.mclip_model is not None and
        search_engine.mclip_tokenizer is not None
    )
    
    # Check Neo4j connection
    neo4j_connected = search_engine.driver is not None
    
    status = "healthy" if (models_loaded and neo4j_connected) else "unhealthy"
    
    return {
        "status": status,
        "models_loaded": models_loaded,
        "neo4j_connected": neo4j_connected
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "5000"))
    uvicorn.run(app, host=host, port=port)