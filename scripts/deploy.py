#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de deploy para GimnTools
Automatiza o processo de publicação e distribuição do pacote
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
import argparse

def run_command(cmd, cwd=None, check=True, capture_output=False):
    """Executa comando e retorna resultado"""
    print(f"🔧 Executando: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            check=check,
            capture_output=capture_output,
            text=True
        )
        if capture_output and result.stdout:
            return result.stdout.strip()
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar comando: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        if check:
            sys.exit(1)
        return None

def get_current_version():
    """Obtém a versão atual do pacote"""
    try:
        # Tenta ler do setup.py
        with open('setup.py', 'r') as f:
            content = f.read()
            match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
        
        # Tenta ler do pyproject.toml
        try:
            import toml
            with open('pyproject.toml', 'r') as f:
                config = toml.load(f)
                return config.get('project', {}).get('version')
        except ImportError:
            pass
        
        # Tenta importar o pacote
        try:
            import GimnTools
            return GimnTools.__version__
        except:
            pass
            
    except Exception as e:
        print(f"⚠️  Não foi possível determinar a versão: {e}")
    
    return "unknown"

def update_version(new_version):
    """Atualiza a versão do pacote"""
    print(f"📝 Atualizando versão para {new_version}...")
    
    # Atualiza setup.py
    if Path('setup.py').exists():
        with open('setup.py', 'r') as f:
            content = f.read()
        
        content = re.sub(
            r'VERSION\s*=\s*["\'][^"\']+["\']',
            f'VERSION = "{new_version}"',
            content
        )
        
        with open('setup.py', 'w') as f:
            f.write(content)
        
        print("✅ setup.py atualizado")
    
    # Atualiza pyproject.toml se não usar setuptools_scm
    if Path('pyproject.toml').exists():
        try:
            import toml
            with open('pyproject.toml', 'r') as f:
                config = toml.load(f)
            
            if 'version' in config.get('project', {}):
                config['project']['version'] = new_version
                with open('pyproject.toml', 'w') as f:
                    toml.dump(config, f)
                print("✅ pyproject.toml atualizado")
        except ImportError:
            print("⚠️  toml não disponível, pyproject.toml não atualizado")

def update_changelog(version, changes=None):
    """Atualiza o CHANGELOG.md"""
    print(f"📝 Atualizando CHANGELOG para versão {version}...")
    
    changelog_path = Path('CHANGELOG.md')
    
    # Cria CHANGELOG se não existir
    if not changelog_path.exists():
        changelog_content = "# Changelog\n\nTodas as mudanças notáveis neste projeto serão documentadas neste arquivo.\n\n"
    else:
        with open(changelog_path, 'r', encoding='utf-8') as f:
            changelog_content = f.read()
    
    # Adiciona nova entrada
    date_str = datetime.now().strftime('%Y-%m-%d')
    new_entry = f"\n## [{version}] - {date_str}\n\n"
    
    if changes:
        for change in changes:
            new_entry += f"- {change}\n"
    else:
        new_entry += "- Atualizações e melhorias diversas\n"
    
    # Insere após o cabeçalho
    lines = changelog_content.split('\n')
    insert_pos = 3  # Após título e descrição
    for i, line in enumerate(lines):
        if line.startswith('## ['):
            insert_pos = i
            break
    
    lines.insert(insert_pos, new_entry)
    
    with open(changelog_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("✅ CHANGELOG.md atualizado")

def run_tests():
    """Executa testes antes do deploy"""
    print("🧪 Executando testes...")
    
    try:
        # Executa testes
        result = run_command("python -m pytest tests/ -v", check=False, capture_output=True)
        if result and result.returncode == 0:
            print("✅ Todos os testes passaram")
            return True
        else:
            print("❌ Alguns testes falharam")
            return False
    except:
        print("⚠️  Pytest não disponível, pulando testes")
        return True

def check_code_quality():
    """Verifica qualidade do código"""
    print("🔍 Verificando qualidade do código...")
    
    checks_passed = True
    
    # Flake8
    try:
        result = run_command("flake8 GimnTools/ --max-line-length=88", 
                           check=False, capture_output=True)
        if result and result.returncode == 0:
            print("✅ Flake8 passou")
        else:
            print("⚠️  Avisos do Flake8 encontrados")
            checks_passed = False
    except:
        print("⚠️  Flake8 não disponível")
    
    # Black
    try:
        result = run_command("black --check GimnTools/", 
                           check=False, capture_output=True)
        if result and result.returncode == 0:
            print("✅ Formatação Black está OK")
        else:
            print("⚠️  Código precisa ser formatado com Black")
            checks_passed = False
    except:
        print("⚠️  Black não disponível")
    
    return checks_passed

def build_package():
    """Compila o pacote"""
    print("🔨 Compilando pacote...")
    
    # Limpa builds anteriores
    run_command("rm -rf build/ dist/ *.egg-info/", check=False)
    
    # Compila
    run_command("python -m build")
    
    # Verifica se arquivos foram criados
    dist_files = list(Path('dist').glob('*'))
    if not dist_files:
        print("❌ Nenhum arquivo gerado em dist/")
        return False
    
    print(f"✅ Pacote compilado: {len(dist_files)} arquivo(s)")
    for file in dist_files:
        print(f"   📦 {file.name}")
    
    return True

def check_package_quality():
    """Verifica qualidade do pacote compilado"""
    print("🔍 Verificando qualidade do pacote...")
    
    try:
        run_command("twine check dist/*")
        print("✅ Pacote passou na verificação do Twine")
        return True
    except:
        print("❌ Pacote falhou na verificação do Twine")
        return False

def create_git_tag(version):
    """Cria tag no git"""
    print(f"🏷️  Criando tag git v{version}...")
    
    try:
        # Verifica se há mudanças não commitadas
        result = run_command("git status --porcelain", capture_output=True)
        if result and result.strip():
            print("⚠️  Há mudanças não commitadas:")
            print(result)
            confirm = input("Continuar mesmo assim? (y/N): ")
            if confirm.lower() != 'y':
                return False
        
        # Cria tag
        run_command(f"git tag -a v{version} -m 'Release v{version}'")
        print("✅ Tag criada")
        
        # Push da tag
        push_tag = input("Fazer push da tag? (y/N): ")
        if push_tag.lower() == 'y':
            run_command("git push origin --tags")
            print("✅ Tag enviada para o repositório")
        
        return True
        
    except:
        print("❌ Erro ao criar tag git")
        return False

def upload_to_pypi(repository='testpypi'):
    """Faz upload para PyPI"""
    print(f"🚀 Fazendo upload para {repository}...")
    
    if repository == 'testpypi':
        cmd = "twine upload --repository testpypi dist/*"
    else:
        # Confirmação adicional para PyPI oficial
        print("⚠️  ATENÇÃO: Você está prestes a publicar no PyPI OFICIAL!")
        print("Esta ação não pode ser desfeita.")
        confirm = input("Confirma a publicação? Digite 'CONFIRMO': ")
        if confirm != 'CONFIRMO':
            print("❌ Upload cancelado")
            return False
        
        cmd = "twine upload dist/*"
    
    try:
        run_command(cmd)
        print(f"✅ Upload para {repository} concluído com sucesso!")
        return True
    except:
        print(f"❌ Erro no upload para {repository}")
        return False

def deploy_workflow(version=None, repository='testpypi', skip_tests=False, skip_quality=False):
    """Executa o workflow completo de deploy"""
    print("🚀 Iniciando processo de deploy...")
    print("=" * 50)
    
    current_version = get_current_version()
    print(f"📋 Versão atual: {current_version}")
    
    if version:
        if version != current_version:
            update_version(version)
            update_changelog(version)
    else:
        version = current_version
    
    print(f"📋 Versão para deploy: {version}")
    
    # Verificações pré-deploy
    if not skip_tests:
        if not run_tests():
            print("❌ Testes falharam. Deploy cancelado.")
            return False
    
    if not skip_quality:
        if not check_code_quality():
            proceed = input("⚠️  Verificações de qualidade falharam. Continuar? (y/N): ")
            if proceed.lower() != 'y':
                print("❌ Deploy cancelado.")
                return False
    
    # Build e verificação
    if not build_package():
        print("❌ Falha na compilação. Deploy cancelado.")
        return False
    
    if not check_package_quality():
        print("❌ Falha na verificação do pacote. Deploy cancelado.")
        return False
    
    # Git tag
    create_git_tag(version)
    
    # Upload
    if upload_to_pypi(repository):
        print()
        print("🎉 Deploy concluído com sucesso!")
        if repository == 'testpypi':
            print("📋 Para testar a instalação:")
            print(f"   pip install -i https://test.pypi.org/simple/ GimnTools=={version}")
        else:
            print("📋 Pacote agora disponível no PyPI:")
            print(f"   pip install GimnTools=={version}")
        return True
    else:
        print("❌ Deploy falhou no upload.")
        return False

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Script de deploy para GimnTools')
    parser.add_argument('--version', type=str, help='Nova versão para deploy')
    parser.add_argument('--repository', choices=['testpypi', 'pypi'], 
                       default='testpypi', help='Repositório de destino')
    parser.add_argument('--skip-tests', action='store_true', 
                       help='Pula execução de testes')
    parser.add_argument('--skip-quality', action='store_true', 
                       help='Pula verificações de qualidade')
    parser.add_argument('--build-only', action='store_true', 
                       help='Apenas compila, não faz upload')
    parser.add_argument('--upload-only', action='store_true', 
                       help='Apenas faz upload (assume que já foi compilado)')
    
    args = parser.parse_args()
    
    # Muda para diretório do projeto
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    os.chdir(project_dir)
    
    print("🚀 GimnTools Deploy Script")
    print("=" * 50)
    
    try:
        if args.build_only:
            # Apenas compila
            if build_package() and check_package_quality():
                print("✅ Build concluído com sucesso!")
            else:
                print("❌ Build falhou.")
                sys.exit(1)
                
        elif args.upload_only:
            # Apenas upload
            if check_package_quality() and upload_to_pypi(args.repository):
                print("✅ Upload concluído com sucesso!")
            else:
                print("❌ Upload falhou.")
                sys.exit(1)
        else:
            # Workflow completo
            success = deploy_workflow(
                version=args.version,
                repository=args.repository,
                skip_tests=args.skip_tests,
                skip_quality=args.skip_quality
            )
            
            if not success:
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n❌ Deploy interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro durante deploy: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
