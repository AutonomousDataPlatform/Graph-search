# Docker로 Graph Search Server 실행하기

## 빠른 시작

### 1. Docker 이미지 빌드
```bash
cd server
docker build -t graph-search-server .
```

### 2. Docker 컨테이너 실행
```bash
docker run -p 5000:5000 graph-search-server
```

서버가 http://localhost:5000 에서 실행됩니다.

## Docker Compose 사용 (권장)

### 1. 빌드 및 실행
```bash
cd server
docker-compose up -d
```

### 2. 로그 확인
```bash
docker-compose logs -f
```

### 3. 중지
```bash
docker-compose down
```

### 4. 재빌드
```bash
docker-compose up -d --build
```

## 환경 변수 설정

`.env` 파일을 수정하거나, `docker-compose.yml`에서 환경 변수를 직접 설정할 수 있습니다.

### .env 파일 예시
```env
NEO4J_URL=bolt://54.249.172.114:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
DEVICE=cpu
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=5000
```

## GPU 지원 (선택사항)

GPU를 사용하려면 Dockerfile을 수정하여 CUDA 버전의 PyTorch를 설치하세요:

```dockerfile
# Dockerfile에서 이 줄을 찾아서
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio --index-url https://download.pytorch.org/whl/cpu

# 다음과 같이 변경
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio --index-url https://download.pytorch.org/whl/cu121
```

그리고 `docker-compose.yml`에 GPU 설정 추가:

```yaml
services:
  graph-search-server:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

그리고 `.env` 파일에서 `DEVICE=cuda`로 변경하세요.

## 포트 변경

다른 포트를 사용하려면 `docker-compose.yml`에서 포트 매핑을 변경하세요:

```yaml
ports:
  - "8080:5000"  # 호스트:컨테이너
```

또는 docker run 명령어 사용 시:

```bash
docker run -p 8080:5000 graph-search-server
```

## 개발 모드

개발 중에는 코드 변경사항을 즉시 반영하기 위해 볼륨 마운트를 사용할 수 있습니다:

```bash
docker run -p 5000:5000 \
  -v $(pwd)/static:/app/static \
  -v $(pwd)/templates:/app/templates \
  -v $(pwd)/search.py:/app/search.py \
  graph-search-server
```

## 문제 해결

### 컨테이너 내부 접속
```bash
docker exec -it graph-search-server bash
```

### 로그 확인
```bash
docker logs graph-search-server
```

### 이미지 삭제 및 재빌드
```bash
docker-compose down
docker rmi graph-search-server
docker-compose up -d --build
```
