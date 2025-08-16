# ================================
# Makefile para GimnTools
# ================================

# Detecta python automaticamente (pode ser sobrescrito: make install PYTHON=python3.11)
PYTHON ?= python3

# Cores para saída
GREEN  := \033[0;32m
RED    := \033[0;31m
YELLOW := \033[1;33m
NC     := \033[0m  # Sem cor

# Alvo padrão
.DEFAULT_GOAL := help

# --------------------------------
# Instalação
# --------------------------------
install: ## Instala o pacote e dependências básicas
	@echo "$(GREEN)Instalando GimnTools...$(NC)"
	$(PYTHON) ./scripts/install.py



install-dev: ## Instala o pacote e dependências básicas
	@echo "$(GREEN)Instalando GimnTools...$(NC)"
	$(PYTHON) ./scripts/install.py --mode development
# --------------------------------
# Limpeza
# --------------------------------
clean: ## Remove arquivos temporários e de build
	@echo "$(YELLOW)Limpando arquivos temporários...$(NC)"
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache

# --------------------------------
# Testes
# --------------------------------
test: ## Executa testes do pacote
	@echo "$(GREEN)Executando testes...$(NC)"
	$(PYTHON) ./GimnTools/tests/tests.py

# --------------------------------
# Empacotamento
# --------------------------------
build: clean ## Gera pacotes (sdist e wheel)
	@echo "$(GREEN)Gerando pacotes para PyPI...$(NC)"
	$(PYTHON) setup.py sdist bdist_wheel

# --------------------------------
# Upload para PyPI
# --------------------------------
upload: build ## Faz upload do pacote para o PyPI
	@echo "$(GREEN)Enviando para PyPI...$(NC)"
	twine upload dist/*

# --------------------------------
# Ajuda (lista os comandos disponíveis)
# --------------------------------
help: ## Mostra esta ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1;33m%-10s\033[0m %s\n", $$1, $$2}'
