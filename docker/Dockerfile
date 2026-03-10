# ===========================================================================
# Multi-stage Dockerfile for LogFileAnalyzer
# Stage 1: Build React frontend (no source maps)
# Stage 2: Compile Python to .pyc (source protection)
# Stage 3: Slim runtime with Nginx + Gunicorn
# ===========================================================================

# ---- Stage 1: Frontend build ----
FROM node:20-alpine AS frontend-build
WORKDIR /build
COPY --from=react-app package.json package-lock.json* ./
RUN npm ci --ignore-scripts 2>/dev/null || npm install
COPY --from=react-app . ./
# Build without source maps so the bundle cannot be reverse-mapped to source
# Clear REACT_APP_BACKEND_URL so the app uses relative URLs (proxied by Nginx)
ENV GENERATE_SOURCEMAP=false \
    REACT_APP_BACKEND_URL=""
RUN npm run build

# ---- Stage 2: Python compilation ----
FROM tensorflow/tensorflow:2.14.0 AS python-compile
WORKDIR /compile

# Install build dependencies for bcrypt
RUN apt-get update && apt-get install -y --no-install-recommends gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy server code + LogFileAnalyzer (server imports from it)
COPY Server/ ./Server/
COPY LogFileAnalyzer/ ./LogFileAnalyzer/

# Install Python dependencies
# Remove distutils-installed blinker to avoid pip conflict
RUN rm -rf /usr/lib/python3/dist-packages/blinker*
COPY Server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Compile all .py to .pyc in-place (-b writes .pyc next to .py)
RUN python -m compileall -b Server/ LogFileAnalyzer/ && \
    # Remove all .py source files — only .pyc remains
    find Server/ LogFileAnalyzer/ -name "*.py" -delete

# ---- Stage 3: Runtime ----
FROM tensorflow/tensorflow:2.14.0 AS runtime
WORKDIR /app

# Install Nginx + runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx curl && \
    rm -rf /var/lib/apt/lists/*

# Re-install Python packages in runtime image
# Remove distutils-installed blinker to avoid pip conflict
RUN rm -rf /usr/lib/python3/dist-packages/blinker*
COPY Server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy compiled Python code (only .pyc, no .py source)
COPY --from=python-compile /compile/Server/ ./Server/
COPY --from=python-compile /compile/LogFileAnalyzer/ ./LogFileAnalyzer/

# Copy built React app into Nginx static directory
COPY --from=frontend-build /build/build/ /usr/share/nginx/html/

# Copy Nginx config
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Copy entrypoint
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create data directory (mountable volume)
RUN mkdir -p /app/data

# Environment defaults
ENV JWT_SECRET=change-me-in-production \
    CORS_ORIGINS="*" \
    USER_DB_PATH=/app/data/users.db \
    DATA_BASE_DIRECTORY=/app/data \
    LOG_FILE_ANALYZER_DIRECTORY=/app/LogFileAnalyzer \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost/ || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
