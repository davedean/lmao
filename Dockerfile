# Dockerfile for LMAO (LM Studio Agent Operator)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the entire LMAO project
COPY . .

# No setup.py/pyproject.toml, just ensure Python can find the modules
# The package uses the standard library only, so no additional dependencies needed

# Create a non-root user for security
RUN useradd -m -u 1000 lmao && chown -R lmao:lmao /app
USER lmao

# Set default environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose a volume for workspace data
VOLUME ["/workspace"]

# Default command - show help
ENTRYPOINT ["python", "-m", "lmao"]
CMD ["--help"]