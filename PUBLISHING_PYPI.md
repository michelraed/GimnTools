# Publicação do GimnToolsV2 no PyPI (passo a passo)

Este procedimento **não realiza o upload automaticamente**. Ao seguir os
passos você publicará `GimnToolsV2` em https://pypi.org, permitindo
`pip install GimnToolsV2`. O nome deve estar disponível no momento real
do envio. O índice normaliza maiúsculas/minúsculas, hífens e sublinhados.

## 0. Antes de publicar — revisão de segurança e direitos

- Confirme que você detém os direitos necessários sobre os módulos, inclusive
  eventuais contribuições de terceiros. O MIT não substitui a permissão dos
  respectivos titulares nem elimina licenças de dependências.
- Verifique que **não há dados de pacientes, segredos, tokens, arquivos
  experimentais privados ou calibragens pessoais** na pasta de publicação.
- Confira nome e e-mail do autor em `pyproject.toml`, `CITATION.cff` e
  `CODE_OF_CONDUCT.md`. O e-mail será metadado público no PyPI.
- Confirme que é aceitável publicar esta versão **0.4.3** (a mesma versão não
  pode ser reutilizada para modificar uma publicação existente).
- A licença MIT permite uso comercial, modificação e redistribuição desde
  que sejam mantidos os avisos de direitos autorais e a licença. **Citar em
  artigos é pedido científico, não obrigação jurídica adicional.**
- Confira precisão técnica e científica antes do anúncio público. Os testes
  computacionais não constituem uma validação do equipamento PET.

## 1. Preparar ambiente Linux/macOS

Na pasta onde está `pyproject.toml`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade build twine pytest
```

No Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip build twine pytest
```

Instale as dependências e rode testes:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/ -q
# Linux com interface virtual instalada, opcional:
xvfb-run -a python -m pytest tests/ -q
```

> Se algum teste não puder rodar por dependência ou display ausente, não
> declare que passou. Resolva ou documente explicitamente o limite.

## 2. Gerar os dois arquivos publicáveis

```bash
# Execute na raiz do projeto, nunca na pasta dist/
rm -rf dist build
python -m build
python -m twine check dist/*
```

No Windows, apague as pastas `dist` e `build` pelo Explorer ou PowerShell
antes de `python -m build`.

Esperado:

```text
dist/
  gimntoolsv2-0.4.3-py3-none-any.whl
  gimntoolsv2-0.4.3.tar.gz
```

**Wheel** (`.whl`) = instalação direta por pip; **sdist** (`.tar.gz`) =
fontes para instalação/build. Ambas devem ser publicadas. Não envie
um ZIP externo contendo a pasta `dist` no lugar desses arquivos.

## 3. Testar a instalação local

```bash
python -m pip install --force-reinstall --no-deps dist/gimntoolsv2-0.4.3-py3-none-any.whl
python -c "from Gimntools.GimnRec import __version__; print(__version__)"
reconstruct_gate --help
reconstruct_petsys --help
```

`--no-deps` é apenas para teste local num ambiente com dependências
instaladas. Para usuários, usar `pip install GimnToolsV2` normalmente.

## 4. Criar conta em TestPyPI e testar upload

1. Registre uma conta em https://test.pypi.org/account/register/ e confirme
   o e-mail. Use 2FA e guarde códigos de recuperação.
2. Crie um token em https://test.pypi.org/manage/account/#api-tokens
   (para o primeiro upload, pode ser necessário escopo da conta).
3. Execute:

```bash
python -m twine upload --repository testpypi dist/*
```

Ao solicitar credenciais, usuário `__token__` e senha **token completo**
começando por `pypi-`. Não inclua credenciais em comandos versionados,
README, histórico do shell ou `git`.

Abra `https://test.pypi.org/project/GimnToolsV2/0.4.3/` para verificar
se a página renderiza corretamente. Para testar o pacote de testes sem
misturar repositórios, primeiro instale dependências via PyPI normal e
instale o pacote TestPyPI sem suas dependências:

```bash
python -m pip install -e .   # dependências somente em ambiente isolado
python -m pip uninstall -y GimnToolsV2
python -m pip install --no-deps --index-url https://test.pypi.org/simple/ GimnToolsV2==0.4.3
```

Não confunda a conta TestPyPI com a conta PyPI de produção. Publicar no
TestPyPI não publica no PyPI principal.

## 5. Publicar no PyPI de produção

1. Registre uma conta em https://pypi.org/account/register/ e confirme o e-mail.
2. Ative autenticação de dois fatores e gere um token pessoal em
   https://pypi.org/manage/account/#api-tokens . Se o projeto ainda não
   existir, o primeiro token pode precisar de escopo de conta inteira;
   depois restrinja tokens ao projeto.
3. **Confirme novamente** nome e versão no PyPI. Se a versão já existir,
   aumente a versão no `pyproject.toml`, `CITATION.cff` e `__version__`
   antes de reconstruir os artefatos.
4. Execute:

```bash
python -m twine upload dist/*
```

Usuário `__token__`, senha `pypi-...` (token completo).

Depois confira:

- https://pypi.org/project/GimnToolsV2/
- https://pypi.org/project/GimnToolsV2/0.4.3/

## 6. Instalar de forma independente após publicação

```bash
python3 -m venv .test_install
source .test_install/bin/activate
python -m pip install GimnToolsV2==0.4.3
python -c "from Gimntools.GimnRec import __version__; print(__version__)"
reconstruct_gate --help
```

No Ubuntu/Debian, para a interface gráfica, instale `python3-tk` também.

## 7. Próximas versões

No PyPI **não é possível sobrescrever arquivos nem reutilizar a mesma
versão**. Para uma correção, incremente `0.4.3` para, por exemplo, `0.4.4`
(em `pyproject.toml`, `src/Gimntools/GimnRec/__init__.py`, `CITATION.cff`
e CHANGELOG), rode testes, remova `dist/`, gere nova build e publique.
Também registre tag Git correspondente (`v0.4.4`).

## 8. Opção mais segura para releases posteriores: Trusted Publishing

A documentação oficial recomenda publicar via **Trusted Publisher**
(GitHub Actions/OIDC), sem guardar tokens estáticos no GitHub. Configure
em PyPI a combinação exata **usuário/organização, repositório, workflow e
ambiente** e publique com `pypa/gh-action-pypi-publish`, concedendo a esse
job `permissions: id-token: write`. A primeira publicação também pode
ser feita por um *pending publisher*, configurado em sua conta PyPI.

Veja: https://docs.pypi.org/trusted-publishers/ e
https://docs.pypi.org/trusted-publishers/using-a-publisher/ .

## 9. Citação com DOI persistente (GitHub + Zenodo)

Publique o código em um repositório GitHub, mantenha `CITATION.cff`,
faça um Release com a tag `v0.4.3`, habilite a integração do GitHub com o
Zenodo e gere um DOI. **Somente depois** adicione o DOI verdadeiro em
`CITATION.cff` e na documentação em uma versão futura. O Zenodo costuma
registrar o CFF no arquivo do release; evite manter `.zenodo.json`
conflitante sem entender sua precedência.

Fontes oficiais:
- https://packaging.python.org/en/latest/tutorials/packaging-projects/
- https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- https://docs.pypi.org/trusted-publishers/
- https://help.zenodo.org/docs/github/describe-software/citation-file/
