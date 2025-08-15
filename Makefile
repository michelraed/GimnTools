# Makefile para GimnTools
# Automação de tarefas de desenvolvimento, build e deploy

.PHONY: help install install-dev test lint format build deploy clean docs

# Configurações
PYTHON := python3
PIP := pip3
PACKAGE_NAME := GimnTools

# Cores para output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Mostra esta mensagem de ajuda
	@echo "$(GREEN)GimnTools - Comandos disponíveis:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## Instala o pacote e dependências básicas
	@echo "$(GREEN)Instalando GimnTools...$(NC)"
	$(PIP) install -e .

install-dev: ## Instala o pacote com dependências de desenvolvimento
	@echo "$(GREEN)Instalando GimnTools para desenvolvimento...$(NC)"
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements-dev.txt
	pre-commit install

install-all: ## Instala todas as dependências (básicas + dev + docs + jupyter)
	@echo "$(GREEN)Instalando todas as dependências...$(NC)"
	$(PIP) install -e ".[all]"
	$(PIP) install -r requirements-dev.txt

test: ## Executa todos os testes
	@echo "$(GREEN)Executando testes...$(NC)"
	pytest tests/ -v --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term

test-fast: ## Executa testes rapidamente (sem coverage)
	@echo "$(GREEN)Executando testes rápidos...$(NC)"
	pytest tests/ -v -x

lint: ## Executa verificação de código (flake8, mypy)
	@echo "$(GREEN)Verificando código...$(NC)"
	flake8 $(PACKAGE_NAME)/
	mypy $(PACKAGE_NAME)/
	pylint $(PACKAGE_NAME)/

format: ## Formata o código (black, isort)
	@echo "$(GREEN)Formatando código...$(NC)"
	black $(PACKAGE_NAME)/ tests/
	isort $(PACKAGE_NAME)/ tests/

format-check: ## Verifica formatação sem modificar
	@echo "$(GREEN)Verificando formatação...$(NC)"
	black --check $(PACKAGE_NAME)/ tests/
	isort --check-only $(PACKAGE_NAME)/ tests/

build: clean ## Compila o pacote para distribuição
	@echo "$(GREEN)Compilando pacote...$(NC)"
	$(PYTHON) -m build

build-wheel: clean ## Compila apenas wheel
	@echo "$(GREEN)Compilando wheel...$(NC)"
	$(PYTHON) -m build --wheel

build-sdist: clean ## Compila apenas source distribution
	@echo "$(GREEN)Compilando source distribution...$(NC)"
	$(PYTHON) -m build --sdist

check: ## Verifica a qualidade do pacote compilado
	@echo "$(GREEN)Verificando pacote...$(NC)"
	twine check dist/*

upload-test: build check ## Faz upload para PyPI de teste
	@echo "$(YELLOW)Fazendo upload para PyPI de teste...$(NC)"
	twine upload --repository testpypi dist/*

upload: build check ## Faz upload para PyPI oficial
	@echo "$(RED)Fazendo upload para PyPI oficial...$(NC)"
	@echo "$(RED)ATENÇÃO: Esta ação irá publicar o pacote publicamente!$(NC)"
	@read -p "Continuar? [y/N] " confirm && [ "$$confirm" = "y" ]
	twine upload dist/*

deploy: upload ## Alias para upload

docs: ## Gera documentação
	@echo "$(GREEN)Gerando documentação...$(NC)"
	cd docs && $(MAKE) html

docs-clean: ## Limpa documentação gerada
	@echo "$(GREEN)Limpando documentação...$(NC)"
	cd docs && $(MAKE) clean

docs-live: ## Inicia servidor de documentação com auto-reload
	@echo "$(GREEN)Iniciando servidor de documentação...$(NC)"
	sphinx-autobuild docs docs/_build/html

clean: ## Limpa arquivos de build e cache
	@echo "$(GREEN)Limpando arquivos temporários...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean docs-clean ## Limpa tudo (build, cache, docs)

install-system-deps: ## Instala dependências do sistema (Ubuntu/Debian)
	@echo "$(GREEN)Instalando dependências do sistema...$(NC)"
	sudo apt-get update
	sudo apt-get install -y python3-dev python3-pip build-essential
	sudo apt-get install -y libfftw3-dev liblapack-dev libatlas-base-dev
	sudo apt-get install -y libgdcm-tools libinsighttoolkit4-dev

profile: ## Executa profiling de performance
	@echo "$(GREEN)Executando profiling...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats examples/performance_test.py
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumtime').print_stats(20)"

benchmark: ## Executa benchmarks
	@echo "$(GREEN)Executando benchmarks...$(NC)"
	$(PYTHON) -m pytest benchmarks/ -v --benchmark-only

version: ## Mostra versão atual
	@echo "$(GREEN)Versão atual:$(NC)"
	$(PYTHON) -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__version__)"

info: ## Mostra informações do ambiente
	@echo "$(GREEN)Informações do ambiente:$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Diretório: $$(pwd)"
	@echo "Pacotes instalados:"
	@$(PIP) list | grep -E "(numpy|scipy|matplotlib|SimpleITK|itk|numba)"

# Comandos para CI/CD
ci-install: ## Instala dependências para CI
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements-dev.txt

ci-test: ## Executa testes para CI
	pytest tests/ -v --cov=$(PACKAGE_NAME) --cov-report=xml --cov-report=term

ci-lint: ## Executa linting para CI
	flake8 $(PACKAGE_NAME)/ --max-line-length=88 --extend-ignore=E203,W503
	black --check $(PACKAGE_NAME)/
	isort --check-only $(PACKAGE_NAME)/

# Comandos de desenvolvimento
dev-setup: install-dev ## Configura ambiente de desenvolvimento
	@echo "$(GREEN)Ambiente de desenvolvimento configurado!$(NC)"
	@echo "Execute 'make test' para verificar se tudo está funcionando."

dev-reset: clean-all install-dev ## Reseta ambiente de desenvolvimento

# Comandos de exemplo
run-examples: ## Executa exemplos da biblioteca
	@echo "$(GREEN)Executando exemplos...$(NC)"
	$(PYTHON) examples/basic_usage.py
	$(PYTHON) examples/reconstruction_example.py
	$(PYTHON) examples/phantom_generation.py

# Para debugging
debug-install: ## Instala em modo debug (verboso)
	$(PIP) install -e . -v

# Mostra ajuda por padrão
.DEFAULT_GOAL := help
