#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de instalação para GimnTools
Automatiza o processo de instalação e configuração do ambiente
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path




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

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    print("🐍 Verificando versão do Python...")
    
    version = sys.version_info
    if version < (3, 8):
        print(f"❌ Python {version.major}.{version.minor} não é suportado")
        print("GimnTools requer Python 3.8 ou superior")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")

def check_pip():
    """Verifica se pip está disponível e atualiza se necessário"""
    print("📦 Verificando pip...")
    
    try:
        import pip
        # Atualiza pip se possível
        run_command(f"{sys.executable} -m pip install --upgrade pip", check=False)
        print("✅ pip verificado e atualizado")
    except ImportError:
        print("❌ pip não encontrado")
        print("Por favor, instale pip antes de continuar")
        sys.exit(1)

def install_system_dependencies():
    """Instala dependências do sistema operacional"""
    print("🔧 Verificando dependências do sistema...")
    
    system = platform.system().lower()
    
    if system == "linux":
        # Detecta distribuição Linux
        try:
            with open("/etc/os-release") as f:
                os_info = f.read().lower()
        except:
            os_info = ""
        
        if "ubuntu" in os_info or "debian" in os_info:
            print("📋 Detectado: Ubuntu/Debian")
            packages = [
                "python3-dev", "python3-pip", "build-essential",
                "libfftw3-dev", "liblapack-dev", "libatlas-base-dev",
                "libgdcm-tools", "libinsighttoolkit4-dev"
            ]
            
            print("ℹ️  Para instalar dependências do sistema, execute:")
            print(f"sudo apt-get update && sudo apt-get install -y {' '.join(packages)}")
            
        elif "fedora" in os_info or "centos" in os_info or "rhel" in os_info:
            print("📋 Detectado: Fedora/CentOS/RHEL")
            packages = [
                "python3-devel", "gcc", "gcc-c++", "make",
                "fftw-devel", "lapack-devel", "atlas-devel"
            ]
            
            print("ℹ️  Para instalar dependências do sistema, execute:")
            print(f"sudo dnf install -y {' '.join(packages)}")
            
    elif system == "darwin":
        print("📋 Detectado: macOS")
        print("ℹ️  Recomenda-se usar Homebrew para instalar dependências:")
        print("brew install fftw")
        
    elif system == "windows":
        print("📋 Detectado: Windows")
        print("ℹ️  Certifique-se de ter Visual Studio Build Tools instalado")
        
    print("✅ Verificação de dependências do sistema concluída")

def create_virtual_environment():
    """Cria ambiente virtual se solicitado"""
    print("🌍 Criando ambiente virtual...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print(f"⚠️  Ambiente virtual já existe em {venv_path}")
        return venv_path
    
    run_command(f"{sys.executable} -m venv venv")
    print(f"✅ Ambiente virtual criado em {venv_path}")
    
    # Instruções para ativação
    system = platform.system().lower()
    if system == "windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    print(f"ℹ️  Para ativar o ambiente virtual, execute: {activate_cmd}")
    return venv_path

def install_package(mode='user', extras=None):
    """Instala o pacote GimnTools"""
    print(f"📦 Instalando GimnTools (modo: {mode})...")
    
    if mode == 'development':
        # Instalação em modo desenvolvimento
        cmd = f"{sys.executable} -m pip install -e ."
        if extras:
            cmd += f"[{extras}]"
    else:
        # Instalação normal
        cmd = f"{sys.executable} -m pip install ."
        if extras:
            cmd += f"[{extras}]"
    
    run_command(cmd)
    
    # Instala dependências adicionais se necessário
    if mode == 'development':
        if Path("requirements-dev.txt").exists():
            run_command(f"{sys.executable} -m pip install -r requirements-dev.txt")
    
    print("✅ GimnTools instalado com sucesso")

def install_optional_dependencies(include_jupyter=False, include_docs=False):
    """Instala dependências opcionais"""
    if include_jupyter:
        print("📓 Instalando dependências do Jupyter...")
        run_command(f"{sys.executable} -m pip install jupyter ipywidgets plotly")
    
    if include_docs:
        print("📚 Instalando dependências de documentação...")
        run_command(f"{sys.executable} -m pip install sphinx sphinx-rtd-theme")

def verify_installation():
    """Verifica se a instalação foi bem-sucedida"""
    print("🧪 Verificando instalação...")
    
    try:
        # Tenta importar a biblioteca
        import GimnTools
        print("✅ Importação bem-sucedida")
        
        # Verifica versão se disponível
        try:
            version = GimnTools.__version__
            print(f"📋 Versão instalada: {version}")
        except AttributeError:
            print("📋 Versão não disponível")
        
        # Testa importação de módulos principais
        try:
            from GimnTools.ImaGIMN.image import image
            from GimnTools.ImaGIMN.sinogramer.sinogramer import Sinogramer
            print("✅ Módulos principais importados com sucesso")
        except ImportError as e:
            print(f"⚠️  Aviso: Erro na importação de módulos: {e}")
        
        try:
            from GimnTools.tests.tests import run_all_tests
            print("Testando a biblioteca")
            run_all_tests ()
        except Exception as e:
            print(f"❌ Erro ao rodar os testes: {e}")


        return True
        
    except ImportError as e:
        print(f"❌ Erro na importação: {e}")
        return False

def setup_development_environment():
    """Configura ambiente de desenvolvimento"""
    print("⚙️  Configurando ambiente de desenvolvimento...")
    
    # Instala pre-commit se disponível
    try:
        run_command(f"{sys.executable} -m pip install pre-commit")
        run_command("pre-commit install")
        print("✅ Pre-commit configurado")
    except:
        print("⚠️  Não foi possível configurar pre-commit")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Script de instalação para GimnTools')
    parser.add_argument('--mode', choices=['user', 'development'], 
                       default='user', help='Modo de instalação')
    parser.add_argument('--venv', action='store_true', 
                       help='Cria ambiente virtual')
    parser.add_argument('--jupyter', action='store_true',
                       help='Instala dependências do Jupyter')
    parser.add_argument('--docs', action='store_true',
                       help='Instala dependências de documentação')
    parser.add_argument('--system-deps', action='store_true',
                       help='Mostra instruções para dependências do sistema')
    parser.add_argument('--extras', type=str,
                       help='Dependências extras para instalar (dev, jupyter, docs, all)')
    
    args = parser.parse_args()
    
    print("🚀 GimnTools Installation Script")
    print("=" * 50)
    
    try:
        # Verificações básicas
        check_python_version()
        check_pip()
        
        if args.system_deps:
            install_system_dependencies()
        
        # Cria ambiente virtual se solicitado
        if args.venv:
            venv_path = create_virtual_environment()
            print(f"ℹ️  Ambiente virtual criado. Ative-o antes de continuar.")
            return
        
        # Muda para diretório do projeto
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        os.chdir(project_dir)
        
        # Instalação do pacote
        install_package(args.mode, args.extras)
        
        # Dependências opcionais
        if args.jupyter:
            install_optional_dependencies(include_jupyter=True)
        
        if args.docs:
            install_optional_dependencies(include_docs=True)
        
        # Configuração de desenvolvimento
        if args.mode == 'development':
            setup_development_environment()
        
        # Verificação final
        if verify_installation():
            print()
            print("🎉 Instalação concluída com sucesso!")
            print()
            print("📋 Próximos passos:")
            if args.mode == 'development':
                print("   - Execute 'make test' para verificar se tudo está funcionando")
                print("   - Execute 'make docs' para gerar documentação")
            print("   - Consulte a documentação para exemplos de uso")
            print("   - Execute 'python -c \"import GimnTools; print('GimnTools funcionando!')\"'")
        else:
            print("❌ Instalação pode ter falhado. Verifique os erros acima.")
            
    except KeyboardInterrupt:
        print("\n❌ Instalação interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro durante instalação: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
