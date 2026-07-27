FROM python:3.12-slim

# Create non-root user (uid 1000 for host-file compatibility)
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin -u 1000 appuser

WORKDIR /app

# Install production dependencies only
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code with correct ownership
COPY --chown=appuser:appuser . .

EXPOSE 9000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import http.client; c=http.client.HTTPConnection('localhost', 9000); c.request('GET', '/health'); r=c.getresponse(); assert r.status == 200, f'health check failed: {r.status}'"

# Drop privileges
USER appuser

CMD ["python", "server.py"]
