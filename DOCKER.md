# Docker Usage

## Building the Image

```bash
# Build with latest tag
docker build -t lmao:latest .

# Or build with custom tag
docker build -t lmao:test .
docker tag lmao:test lmao:latest  # Tag as latest if needed
```

## Running with Docker

### Basic Usage

```bash
# Show help
docker run --rm lmao:latest --help

# Run with workspace mounted (recommended)
docker run --rm -v $(pwd):/workspace -w /workspace lmao:latest "describe this repo"
```

### With LM Studio

```bash
# Assuming LM Studio is running on host at localhost:1234
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  -e LM_STUDIO_URL=http://host.docker.internal:1234/v1/chat/completions \
  -e LM_STUDIO_MODEL=qwen3-4b-instruct \
  lmao:latest "analyze this codebase"
```

### With OpenRouter

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  -e OPENROUTER_API_KEY=your_api_key_here \
  lmao:latest --provider openrouter --model openai/gpt-4o-mini "summarize this project"
```

### Using Docker Compose

```bash
# Build and run with compose
docker-compose build
docker-compose run --rm lmao

# Or with a specific command
docker-compose run --rm lmao "help me understand this repository"
```

### Headless Mode (Non-interactive)

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  -e LM_STUDIO_URL=http://host.docker.internal:1234/v1/chat/completions \
  -e LM_STUDIO_MODEL=qwen3-4b-instruct \
  lmao:latest --headless "list all Python files in this project"
```

### Configuration

You can mount a config directory:

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -v ~/.config/agents:/config \
  -w /workspace \
  lmao:latest --config /config/lmao.conf "process these files"
```

### Environment Variables

- `LM_STUDIO_URL`: LM Studio endpoint URL
- `LM_STUDIO_MODEL`: LM Studio model name
- `OPENROUTER_API_KEY`: OpenRouter API key
- `PYTHONPATH`: Set to `/app` by default
- `PYTHONUNBUFFERED`: Set to `1` by default for better output

### Security Notes

- The container runs as a non-root user (`lmao:1000`)
- All file operations are confined to the mounted workspace
- No special privileges are required
- The image uses Python 3.11-slim for minimal footprint

### Troubleshooting

1. **"unable to pull lmao:latest"**: This means the image isn't tagged as `latest`. Run `docker tag lmao:test lmao:latest` or build directly with `docker build -t lmao:latest .`
2. **502 Bad Gateway**: Ensure your LM Studio or OpenRouter endpoint is accessible
3. **Permission denied**: Check that your workspace directory has proper permissions
4. **Module not found**: Ensure you're running from the correct working directory