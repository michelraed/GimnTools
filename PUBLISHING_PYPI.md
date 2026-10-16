# Publicação do GimnTools 2.0.0 no PyPI

O projeto **GimnTools** já existe em https://pypi.org/project/GimnTools/ .
A conta mantenedora exibida no PyPI é `michelraed2`.
Para publicar a versão 2.0.0, entre com uma conta com permissão de mantenedor/owner
do projeto. A versão 2.0.0 é uma atualização principal que quebra compatibilidade
com a API `GimnTools.ImaGIMN` da distribuição 1.0.0.0.

## Publicar com GitHub Trusted Publishing (sem tokens compartilhados)

1. Confirme que a branch `main` de `michelraed/GimnTools` contém
   `pyproject.toml` com `name = "GimnTools"` e `version = "2.0.0"`.
2. Na conta PyPI mantenedora: Your projects → GimnTools → Publishing →
   Add a new trusted publisher (GitHub).
3. Configure: Owner `michelraed`; Repository `GimnTools`;
   Workflow filename `publish.yml`; Environment `pypi`.
4. No GitHub, Settings → Environments, crie `pypi` com aprovação manual
   de mantenedor para publicação em produção.
5. Depois de revisar o código, publique uma GitHub Release com tag `v2.0.0`.
   O workflow `.github/workflows/publish.yml` gera wheel e sdist,
   verifica metadados e publica no PyPI com OIDC, sem token permanente.

Não compartilhe senhas ou tokens em chats, issues, repositórios ou logs.

## Alternativa: upload manual na sua máquina

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

No Twine, informe `__token__` como usuário e o token PyPI como senha.
Somente a conta com acesso ao projeto GimnTools pode fazer upload.
A publicação no PyPI não pode sobrescrever arquivos já enviados sob a mesma versão.

## Instalação após publicação

```bash
python -m pip install --upgrade GimnTools==2.0.0
python -c "import Gimntools; print(Gimntools.__version__)"
```

Aprovação de testes de software não é equivalente à validação experimental
de um PET específico.
