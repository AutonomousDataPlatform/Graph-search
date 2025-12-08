// Global variables
let filterOptions = {};
let currentFilters = {
    datasets: [],  // Will be populated from server
    weather: [],
    time_of_day: [],
    road_condition: [],
    traffic_density: [],
    pedestrian_density: [],
    object_filters: {}
};

// Image search variables
let currentSearchMode = 'text';
let uploadedImageFile = null;

// Pagination state
let currentPage = 1;
let hasMoreResults = false;
let lastSearchParams = null;  // Store last search params for "load more"


// Global error surfacing
window.addEventListener('error', function(e) {
    try { showStatus(`JS Error: ${e.message}`, 'error'); } catch(_) {}
    console.error('Global error:', e.message, e.error);
});
window.addEventListener('unhandledrejection', function(e) {
    try { showStatus(`Unhandled: ${e.reason}`, 'error'); } catch(_) {}
    console.error('Unhandled promise rejection:', e.reason);
});

// Optimized weights (fixed based on evaluation results)
// alpha=1.0 (image), beta=0.0 (CLIP text), gamma=0.0 (SentenceTransformer)
const OPTIMIZED_ALPHA = 1.0;
const OPTIMIZED_BETA = 0.0;
const OPTIMIZED_GAMMA = 0.0;

// Filter management
async function loadFilterOptions() {
    try {
        const url = `${window.location.origin}/filters/options`;
        const response = await fetch(url);
        let data = null;
        try {
            data = await response.json();
        } catch (parseErr) {
            const text = await response.text().catch(() => '');
            console.warn('Non-JSON filters/options response:', text.slice(0,200));
        }
        if (!response.ok) {
            throw new Error(`HTTP ${response.status} ${response.statusText}`);
        }
        if (!data || typeof data !== 'object') {
            data = { weather: [], time_of_day: [], road_condition: [], traffic_density: [], pedestrian_density: [] };
        }
        filterOptions = data;
        if (data && data.error) {
            console.warn('Filter options error:', data.error);
            showStatus('Filter options loaded with warnings', 'error');
        }
        ensureFilterContainersAndRender();
    } catch (error) {
        console.error('Failed to load filter options:', error);
        try {
            // Render empty groups so UI is still usable
            filterOptions = { weather: [], time_of_day: [], road_condition: [], traffic_density: [], pedestrian_density: [] };
            ensureFilterContainersAndRender();
        } catch(_) {}
        ['weatherFilters','timeFilters','roadFilters','trafficFilters','pedestrianFilters'].forEach(id => {
            const el = document.getElementById(id);
            if (el && !el.innerHTML) el.innerHTML = `<div style="font-size: 12px; color: #c62828;">Failed to load: ${String(error).slice(0,200)}</div>`;
        });
        try { showStatus(`Filters load failed: ${String(error)}`, 'error'); } catch(_) {}
    }
}

function ensureFilterContainersAndRender(retries = 10) {
    const ids = ['weatherFilters','timeFilters','roadFilters','trafficFilters','pedestrianFilters'];
    const allReady = ids.every(id => document.getElementById(id));
    if (allReady) {
        renderFilterOptions();
    } else if (retries > 0) {
        setTimeout(() => ensureFilterContainersAndRender(retries - 1), 100);
    } else {
        console.warn('Filter containers not ready after retries');
    }
}

// Load object categories and render object filter controls
async function loadObjectCategories() {
    try {
        const response = await fetch('/search/categories');
        const data = await response.json();
        const categories = data.categories || [];
        // Sort alphabetically
        categories.sort();
        renderObjectFilters(categories);
    } catch (error) {
        console.error('Failed to load object categories:', error);
        const container = document.getElementById('objectFilters');
        if (container) {
            container.innerHTML = '<div style="font-size: 12px; color: #999;">No categories available</div>';
        }
    }
}

// Icon mapping by value
const iconMap = {
    // Weather
    'clear': '☀️',
    'overcast': '☁️',
    'rainy': '🌧️',
    'snowy': '🌨️',
    'partly cloudy': '⛅',
    'foggy': '🌫️',
    'undefined': '❓',
    // Time of day
    'daytime': '☀️',
    'night': '🌙',
    'dawn/dusk': '🌅',
    // Road condition
    'dry': '☀️',
    'wet': '💧',
    // Traffic density
    'none': '🚫',
    'low': '🚗',
    'medium': '🚗🚙',
    'high': '🚗🚙🚕',
};

// Get icon for a specific filter type and value
function getIcon(filterType, value) {
    if (filterType === 'pedestrian_density') {
        const pedestrianIcons = {
            'none': '🚫',
            'low': '🚶',
            'medium': '🚶🚶',
            'high': '🚶🚶🚶'
        };
        return pedestrianIcons[value.toLowerCase()] || '📌';
    }
    return iconMap[value.toLowerCase()] || '📌';
}

// Render checkbox list for a filter group into a container
function renderFilterGroup(filterType, containerId) {
    try {
        const container = document.getElementById(containerId);
        if (!container) return;
        const options = (filterOptions && Array.isArray(filterOptions[filterType])) ? filterOptions[filterType] : [];
        if (!options || options.length === 0) {
            container.innerHTML = '<div style="font-size: 12px; color: #999;">No data available</div>';
            return;
        }

        // Filter out 'undefined' values
        const validOptions = options.filter(option =>
            option && option.toLowerCase() !== 'undefined'
        );

        if (validOptions.length === 0) {
            container.innerHTML = '<div style="font-size: 12px; color: #999;">No data available</div>';
            return;
        }

        let html = '';
        validOptions.forEach((option) => {
            const icon = getIcon(filterType, option);
            const id = `${filterType}_${option.replace(/\//g, '_').replace(/\s+/g, '_')}`;
            html += `
                <div class="filter-option">
                    <input type="checkbox" id="${id}" value="${option}" onchange="updateFilters()">
                    <label for="${id}">${icon} ${option}</label>
                </div>
            `;
        });
        container.innerHTML = html;
    } catch (e) {
        console.error('renderFilterGroup failed for', filterType, e);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = '<div style="font-size: 12px; color: #c62828;">Render failed</div>';
        }
    }
}

function renderFilterOptions() {
    // Render dataset filters (all checked by default)
    renderDatasetFilters();
    // Render all other filter groups
    renderFilterGroup('weather', 'weatherFilters');
    renderFilterGroup('time_of_day', 'timeFilters');
    renderFilterGroup('road_condition', 'roadFilters');
    renderFilterGroup('traffic_density', 'trafficFilters');
    renderFilterGroup('pedestrian_density', 'pedestrianFilters');
}

function renderDatasetFilters() {
    const container = document.getElementById('datasetFilters');
    if (!container) return;

    const datasets = filterOptions.datasets || [];
    if (datasets.length === 0) {
        container.innerHTML = '<div style="font-size: 12px; color: #999;">No datasets available</div>';
        return;
    }

    // Initialize currentFilters.datasets with all datasets (all checked by default)
    currentFilters.datasets = [...datasets];

    let html = '';
    datasets.forEach(dataset => {
        const id = `dataset_${dataset.toLowerCase().replace(/[^a-z0-9]/g, '_')}`;
        html += `
            <div class="filter-option">
                <input type="checkbox" id="${id}" value="${dataset}" checked onchange="updateFilters()">
                <label for="${id}">${dataset}</label>
            </div>
        `;
    });
    container.innerHTML = html;
}

function renderObjectFilters(categories) {
    const select = document.getElementById('objectCategorySelect');
    if (!select) return;
    select.innerHTML = '';
    if (!categories || categories.length === 0) {
        select.innerHTML = '<option value="">No categories</option>';
        return;
    }
    categories.forEach(category => {
        const opt = document.createElement('option');
        opt.value = category;
        opt.textContent = category;
        select.appendChild(opt);
    });
    renderSelectedObjectFilters();
}

function renderSelectedObjectFilters() {
    const container = document.getElementById('selectedObjects');
    if (!container) return;
    const entries = Object.entries(currentFilters.object_filters || {});
    if (entries.length === 0) {
        container.innerHTML = '<div style="font-size: 12px; color: #999;">No object filters selected</div>';
        return;
    }
    container.innerHTML = '';
    entries.forEach(([cat, minVal]) => {
        const chip = document.createElement('div');
        chip.style.display = 'flex';
        chip.style.alignItems = 'center';
        chip.style.gap = '6px';
        chip.style.padding = '6px 8px';
        chip.style.border = '1px solid #e0e0e0';
        chip.style.borderRadius = '16px';
        chip.style.background = '#fafafa';

        const label = document.createElement('span');
        label.textContent = cat;

        const input = document.createElement('input');
        input.type = 'number';
        input.min = '1';
        input.step = '1';
        input.value = String(minVal);
        input.style.width = '70px';
        input.oninput = () => {
            const v = parseInt(input.value || '1', 10) || 1;
            currentFilters.object_filters[cat] = v;
        };

        const removeBtn = document.createElement('button');
        removeBtn.textContent = '✖';
        removeBtn.className = 'search-button';
        removeBtn.style.padding = '4px 8px';
        removeBtn.style.fontSize = '12px';
        removeBtn.style.background = '#e57373';
        removeBtn.onclick = () => removeObjectFilter(cat);

        chip.appendChild(label);
        chip.appendChild(input);
        chip.appendChild(removeBtn);
        container.appendChild(chip);
    });
}

function addObjectFilter() {
    const select = document.getElementById('objectCategorySelect');
    const minInput = document.getElementById('objectMinInput');
    if (!select || !minInput) return;
    const cat = select.value;
    const minVal = parseInt(minInput.value || '1', 10) || 1;
    if (!cat) return;
    currentFilters.object_filters = currentFilters.object_filters || {};
    currentFilters.object_filters[cat] = minVal;
    renderSelectedObjectFilters();
}

function removeObjectFilter(category) {
    if (!currentFilters.object_filters) return;
    delete currentFilters.object_filters[category];
    renderSelectedObjectFilters();
}

function updateFilters() {
    // Update dataset filters dynamically
    currentFilters.datasets = [];
    const datasetContainer = document.getElementById('datasetFilters');
    if (datasetContainer) {
        const checkboxes = datasetContainer.querySelectorAll('input[type="checkbox"]:checked');
        checkboxes.forEach(cb => {
            currentFilters.datasets.push(cb.value);
        });
    }

    // Update other filters
    ['weather', 'time_of_day', 'road_condition', 'traffic_density', 'pedestrian_density'].forEach(filterType => {
        currentFilters[filterType] = [];
        const container = document.getElementById(`${filterType === 'time_of_day' ? 'time' : filterType === 'road_condition' ? 'road' : filterType === 'traffic_density' ? 'traffic' : filterType === 'pedestrian_density' ? 'pedestrian' : filterType}Filters`);
        const checkboxes = container.querySelectorAll('input[type="checkbox"]:checked');
        checkboxes.forEach(cb => {
            currentFilters[filterType].push(cb.value);
        });
    });

    // Object filters are managed via add/remove handlers; nothing to do here.
}


function resetFilters() {
    // Reset all filter checkboxes dynamically
    document.querySelectorAll('#filterContent input[type="checkbox"]').forEach(cb => {
        // Dataset checkboxes are checked by default, others unchecked
        cb.checked = cb.closest('#datasetFilters') !== null;
    });

    // Reset object filters selection
    currentFilters.object_filters = {};
    renderSelectedObjectFilters();

    updateFilters();
}

function toggleFilters() {
    const content = document.getElementById('filterContent');
    const icon = document.getElementById('filterToggle');

    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.textContent = '▲';
        // In case rendering was attempted before containers existed
        ensureFilterContainersAndRender();
    } else {
        content.style.display = 'none';
        icon.textContent = '▼';
    }
}

// Search functionality
async function search(loadMore = false) {
    const query = document.getElementById('searchQuery').value.trim();
    if (!query) {
        showStatus('Searching by filters only...', 'info');
    }

    // Reset pagination for new search
    if (!loadMore) {
        currentPage = 1;
        document.getElementById('results').innerHTML = '<div class="loading">⏳ 검색 중...</div>';
    }

    const searchButton = event?.target || document.querySelector('button[onclick="search()"]');
    if (searchButton) {
        searchButton.disabled = true;
        searchButton.textContent = '⏳ Searching...';
    }

    try {
        showStatus('Searching...', 'info');

        // Use FormData for unified /search endpoint
        const formData = new FormData();
        formData.append('query', query);
        formData.append('alpha', OPTIMIZED_ALPHA);
        formData.append('beta', OPTIMIZED_BETA);
        formData.append('gamma', OPTIMIZED_GAMMA);
        formData.append('limit', '100');
        formData.append('page', currentPage);
        formData.append('datasets', JSON.stringify(currentFilters.datasets));
        formData.append('object_filters', JSON.stringify(currentFilters.object_filters));
        formData.append('weather', JSON.stringify(currentFilters.weather));
        formData.append('time_of_day', JSON.stringify(currentFilters.time_of_day));
        formData.append('road_condition', JSON.stringify(currentFilters.road_condition));
        formData.append('traffic_density', JSON.stringify(currentFilters.traffic_density));
        formData.append('pedestrian_density', JSON.stringify(currentFilters.pedestrian_density));

        // Store search params for load more
        lastSearchParams = { type: 'text', query };

        const response = await fetch('/search', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        hasMoreResults = data.has_more;

        displayResults(data.results, loadMore);
        showStatus(`Found ${data.results.length} results (page ${currentPage})`, 'info');

    } catch (error) {
        console.error('Search error:', error);
        showStatus(`Search failed: ${error.message}`, 'error');
    } finally {
        if (searchButton) {
            searchButton.disabled = false;
            searchButton.textContent = '🔍 Search';
        }
    }
}

function displayResults(results, append = false) {
    const resultsDiv = document.getElementById('results');

    // Remove existing load more button if exists
    const existingLoadMore = document.getElementById('loadMoreContainer');
    if (existingLoadMore) {
        existingLoadMore.remove();
    }

    if (results.length === 0 && !append) {
        // No results - check if there are more pages to search
        if (hasMoreResults) {
            resultsDiv.innerHTML = `
                <div class="no-results-container">
                    <div class="loading">현재 페이지에 필터 조건에 맞는 결과가 없습니다.</div>
                    <div id="loadMoreContainer" class="load-more-container">
                        <button class="search-button load-more-btn" onclick="loadMoreResults()">🔍 계속 검색</button>
                    </div>
                </div>
            `;
        } else {
            resultsDiv.innerHTML = '<div class="loading">No results found. Try adjusting your search query or filters.</div>';
        }
        return;
    }

    const resultsHtml = results.map(result => {
        let datasetClass = 'dataset-unknown';
        if (result.dataset === 'KITTI') {
            datasetClass = 'dataset-kitti';
        } else if (result.dataset === 'BDD100k') {
            datasetClass = 'dataset-bdd';
        } else if (result.dataset === 'nuScenes') {
            datasetClass = 'dataset-nuscenes';
        } else if (result.dataset === 'CADC') {
            datasetClass = 'dataset-cadc';
        } else if (result.dataset === 'Waymo') {
            datasetClass = 'dataset-waymo';
        }
        return `
            <div class="result-item">
                <div class="result-image-wrap">
                    <img src="${result.image_url}" alt="${result.dataset} Image" class="result-image"
                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzIwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2Y1ZjVmNSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgYXZhaWxhYmxlPC90ZXh0Pjwvc3ZnPg=='" />
                </div>
                <div class="result-content">
                    <span class="result-dataset ${datasetClass}">${result.dataset}</span>
					<div class="result-nodeid">🆔 ${result.node_id}</div>
                    <div class="result-score">Score: ${result.score.toFixed(3)}</div>
                    <div class="result-scores">
                        Img: ${result.scores.image.toFixed(3)} |
                        Txt: ${result.scores.text ? result.scores.text.toFixed(3) : '0.000'}
                    </div>
                    <div class="result-filepath">📁 ${result.filepath}</div>
                    <div class="result-meta">
                        <div>☀️ ${result.environment.weather} | 🕐 ${result.environment.time_of_day}</div>
                        <div>🛣️ ${result.environment.road_condition}</div>
                        <div>🚦 Traffic: ${result.context.traffic_density} | 🚶 Pedestrians: ${result.context.pedestrian_density}</div>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    if (append) {
        resultsDiv.insertAdjacentHTML('beforeend', resultsHtml);
    } else {
        resultsDiv.innerHTML = resultsHtml;
    }

    // Add load more button if there are more results
    if (hasMoreResults) {
        const loadMoreHtml = `
            <div id="loadMoreContainer" class="load-more-container">
                <button class="search-button load-more-btn" onclick="loadMoreResults()">🔍 계속 검색</button>
            </div>
        `;
        resultsDiv.insertAdjacentHTML('beforeend', loadMoreHtml);
    }
}

// Load more results
async function loadMoreResults() {
    currentPage++;

    if (lastSearchParams?.type === 'image') {
        await searchByImage(true);
    } else {
        await search(true);
    }
}


function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;

    if (type === 'info') {
        setTimeout(() => {
            statusDiv.innerHTML = '';
        }, 3000);
    }
}

// Search mode management
function setSearchMode(mode) {
    currentSearchMode = mode;

    // 탭 버튼 활성화
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // 섹션 표시/숨김
    document.getElementById('textSearchSection').style.display = mode === 'text' ? 'block' : 'none';
    document.getElementById('imageSearchSection').style.display = mode === 'image' ? 'block' : 'none';

    // 이전 결과는 유지 (사용자가 비교 가능하도록)
    // document.getElementById('results').innerHTML = '';

    // 상태 메시지만 초기화
    document.getElementById('status').innerHTML = '';

    // 텍스트 검색 입력 초기화
    if (mode === 'text') {
        document.getElementById('searchQuery').value = '';
    }

    // 이미지 검색 입력 초기화
    if (mode === 'image') {
        document.getElementById('imageTextQuery').value = '';
        removeImage();
    }

    // 필터 상태를 UI와 동기화 (중요!)
    // 탭 전환 시 현재 UI에 체크된 필터만 currentFilters에 반영
    updateFilters();
}

// Image upload functionality
function setupImageUpload() {
    const uploadZone = document.getElementById('imageUploadZone');
    const fileInput = document.getElementById('imageFileInput');
    
    if (uploadZone && fileInput) {
        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('dragover', handleDragOver);
        uploadZone.addEventListener('dragleave', handleDragLeave);
        uploadZone.addEventListener('drop', handleDrop);
        fileInput.addEventListener('change', (e) => handleFileSelect(e));
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect({ target: { files } });
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processImageFile(file);
    }
}

function processImageFile(file) {
    // 파일 검증
    if (!file.type.startsWith('image/')) {
        showStatus('이미지 파일만 업로드 가능합니다.', 'error');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB
        showStatus('파일 크기는 10MB를 초과할 수 없습니다.', 'error');
        return;
    }
    
    // 파일 저장
    uploadedImageFile = file;
    
    // 미리보기 표시
    const reader = new FileReader();
    reader.onload = function(e) {
        showImagePreview(file, e.target.result);
    };
    reader.readAsDataURL(file);
}

function showImagePreview(file, dataUrl) {
    const previewDiv = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImage');
    const nameDiv = document.getElementById('imageName');
    const sizeDiv = document.getElementById('imageSize');
    
    previewImg.src = dataUrl;
    nameDiv.textContent = file.name;
    sizeDiv.textContent = formatFileSize(file.size);
    previewDiv.style.display = 'flex';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function removeImage() {
    uploadedImageFile = null;
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('imageFileInput').value = '';
}

// Optimized hybrid weights (fixed based on evaluation results)
const OPTIMIZED_IMAGE_WEIGHT = 1.0;
const OPTIMIZED_TEXT_WEIGHT = 0.0;

// Search functions
async function searchByImage(loadMore = false) {
    if (!uploadedImageFile) {
        showStatus('이미지를 먼저 업로드해주세요.', 'error');
        return;
    }

    // Reset pagination for new search
    if (!loadMore) {
        currentPage = 1;
        document.getElementById('results').innerHTML = '<div class="loading">⏳ 검색 중...</div>';
    }

    const searchButton = event?.target || document.querySelector('button[onclick="searchByImage()"]');
    if (searchButton) {
        searchButton.disabled = true;
        searchButton.textContent = '⏳ 검색 중...';
    }

    try {
        const textQuery = document.getElementById('imageTextQuery').value.trim();
        const isHybrid = textQuery.length > 0;

        showStatus(isHybrid ? '하이브리드 검색 중...' : '이미지 검색 중...', 'info');

        // Use unified /search endpoint
        const formData = new FormData();
        formData.append('image', uploadedImageFile);
        formData.append('query', textQuery);
        formData.append('image_weight', OPTIMIZED_IMAGE_WEIGHT);
        formData.append('text_weight', OPTIMIZED_TEXT_WEIGHT);
        formData.append('limit', '100');
        formData.append('page', currentPage);

        formData.append('datasets', JSON.stringify(currentFilters.datasets));
        formData.append('object_filters', JSON.stringify(currentFilters.object_filters));
        formData.append('weather', JSON.stringify(currentFilters.weather));
        formData.append('time_of_day', JSON.stringify(currentFilters.time_of_day));
        formData.append('road_condition', JSON.stringify(currentFilters.road_condition));
        formData.append('traffic_density', JSON.stringify(currentFilters.traffic_density));
        formData.append('pedestrian_density', JSON.stringify(currentFilters.pedestrian_density));

        // Store search params for load more
        lastSearchParams = { type: 'image', query: textQuery };

        const response = await fetch('/search', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        hasMoreResults = data.has_more;

        displayResults(data.results, loadMore);
        showStatus(`${isHybrid ? '하이브리드' : '이미지'} 검색 완료: ${data.results.length}개 결과 (page ${currentPage})`, 'info');

    } catch (error) {
        console.error('Search error:', error);
        showStatus(`검색 실패: ${error.message}`, 'error');
    } finally {
        if (searchButton) {
            searchButton.disabled = false;
            searchButton.textContent = '🔍 검색';
        }
    }
}


// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadFilterOptions();
    loadObjectCategories();
    setupImageUpload();
    // Allow DOM to settle before first update/render
    setTimeout(() => {
        ensureFilterContainersAndRender();
        updateFilters();
    }, 0);
});

// Allow Enter key to trigger search
document.getElementById('searchQuery').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        search();
    }
});