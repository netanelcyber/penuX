#!/usr/bin/env python3
"""
PenuX LDAP - Complete Automation Script
Automates: Local deployment → GitHub Pages → Vercel API → DNS Setup
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.ENDC}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def input_prompt(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default"""
    if default:
        full_prompt = f"{Colors.CYAN}{prompt} [{default}]: {Colors.ENDC}"
    else:
        full_prompt = f"{Colors.CYAN}{prompt}: {Colors.ENDC}"

    response = input(full_prompt).strip()
    return response if response else default

def run_command(cmd: str, shell: bool = False) -> bool:
    """Run shell command and return success status"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, capture_output=False)
        else:
            result = subprocess.run(cmd.split(), capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print_error(f"Failed to run command: {e}")
        return False

def run_command_silent(cmd: str, shell: bool = False) -> tuple:
    """Run command silently and return output"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def check_docker() -> bool:
    """Check if Docker is installed and running"""
    print_info("Checking Docker installation...")
    success, _ = run_command_silent("docker --version")
    if not success:
        print_error("Docker not installed")
        print_info("Download: https://www.docker.com/products/docker-desktop")
        return False
    print_success("Docker installed")

    success, _ = run_command_silent("docker ps")
    if not success:
        print_error("Docker daemon not running")
        print_info("Please start Docker Desktop")
        return False
    print_success("Docker daemon running")
    return True

def check_git() -> bool:
    """Check if Git is installed"""
    print_info("Checking Git installation...")
    success, _ = run_command_silent("git --version")
    if not success:
        print_error("Git not installed")
        return False
    print_success("Git installed")
    return True

def check_node() -> bool:
    """Check if Node.js is installed"""
    print_info("Checking Node.js installation...")
    success, _ = run_command_silent("node --version")
    if not success:
        print_warning("Node.js not installed (needed for API deployment)")
        return False
    print_success("Node.js installed")
    return True

def phase1_local_deployment(repo_path: str) -> bool:
    """Phase 1: Deploy OpenLDAP locally"""
    print_header("Phase 1: Local OpenLDAP Deployment")

    ldap_path = os.path.join(repo_path, "services", "openldap")
    if not os.path.exists(ldap_path):
        print_error(f"OpenLDAP directory not found: {ldap_path}")
        return False

    print_info(f"Using OpenLDAP directory: {ldap_path}")

    # Create .env file
    os.chdir(ldap_path)
    if not os.path.exists(".env"):
        print_info("Creating .env file...")
        run_command("cp .env.example .env", shell=True)
        print_success(".env file created")

    # Create certs directory
    os.makedirs("certs", exist_ok=True)

    # Start Docker containers
    print_info("Starting Docker containers...")
    if run_command("docker-compose up -d", shell=True) or run_command("docker compose up -d", shell=True):
        print_success("Docker containers started")
    else:
        print_error("Failed to start Docker containers")
        return False

    # Wait for initialization
    print_info("Waiting for services to initialize (40 seconds)...")
    for i in range(40, 0, -1):
        sys.stdout.write(f"\r  {i} seconds remaining...")
        sys.stdout.flush()
        time.sleep(1)
    print()

    print_success("OpenLDAP services initialized")
    print_info("Access at: http://localhost")
    print_info("Admin DN: cn=admin,dc=penux,dc=uk")
    print_info("Password: admin123")

    return True

def phase2_github_pages(repo_path: str, github_user: str) -> bool:
    """Phase 2: Set up GitHub Pages"""
    print_header("Phase 2: GitHub Pages Setup")

    pages_repo = f"{github_user}.github.io"
    pages_path = f"/tmp/{pages_repo}"

    print_info(f"Creating local GitHub Pages repository: {pages_repo}")
    print_warning("Make sure you've created this repository on GitHub first!")
    print_info("Go to: https://github.com/new")
    print_info(f"Create repository named: {pages_repo}")

    input_prompt("Press Enter when repository is created on GitHub")

    # Clone the repository
    print_info(f"Cloning repository...")
    if run_command(f"git clone https://github.com/{github_user}/{pages_repo} {pages_path}", shell=True):
        print_success(f"Repository cloned to {pages_path}")
    else:
        print_error("Failed to clone repository")
        print_info(f"Please clone manually: git clone https://github.com/{github_user}/{pages_repo}")
        return False

    # Copy web interface
    web_file = os.path.join(repo_path, "services", "openldap", "web", "index.html")
    if not os.path.exists(web_file):
        print_error(f"Web interface not found: {web_file}")
        return False

    print_info("Copying web interface...")
    os.chdir(pages_path)
    run_command(f"cp {web_file} ./index.html", shell=True)

    # Commit and push
    print_info("Committing and pushing to GitHub...")
    run_command("git add index.html", shell=True)
    run_command('git commit -m "Add LDAP directory web interface"', shell=True)
    if run_command("git push origin main", shell=True):
        print_success("Web interface pushed to GitHub Pages")
    else:
        print_error("Failed to push to GitHub")
        return False

    print_info("Enabling GitHub Pages...")
    print_warning("Please complete these steps manually:")
    print(f"  1. Go to: https://github.com/{github_user}/{pages_repo}/settings/pages")
    print("  2. Select Source: main / (root)")
    print("  3. Wait ~1 minute for deployment")
    print(f"  4. Access at: https://{pages_repo}")

    input_prompt("Press Enter when GitHub Pages is enabled and working")

    print_success("GitHub Pages configured")
    return True

def phase3_vercel_api(repo_path: str) -> bool:
    """Phase 3: Deploy API to Vercel"""
    print_header("Phase 3: Vercel API Deployment")

    # Check if Vercel CLI is installed
    success, _ = run_command_silent("vercel --version")
    if not success:
        print_warning("Vercel CLI not installed, installing...")
        run_command("npm install -g vercel", shell=True)

    api_path = os.path.join(repo_path, "services", "openldap", "api")
    if not os.path.exists(api_path):
        print_error(f"API directory not found: {api_path}")
        return False

    os.chdir(api_path)

    # Install dependencies
    print_info("Installing API dependencies...")
    if run_command("npm install", shell=True):
        print_success("Dependencies installed")
    else:
        print_error("Failed to install dependencies")
        return False

    # Deploy to Vercel
    print_info("Deploying API to Vercel...")
    print_warning("Make sure you're logged into Vercel (vercel login)")
    if run_command("vercel --prod", shell=True):
        print_success("API deployed to Vercel")
    else:
        print_error("Failed to deploy to Vercel")
        return False

    vercel_url = input_prompt("Enter your Vercel API URL (e.g., your-api.vercel.app)")

    # Set environment variables
    print_info("Setting environment variables in Vercel...")

    local_ip = input_prompt("Enter your local machine IP address for LDAP_HOST (e.g., 192.168.1.100)")
    ldap_host = f"ldap://{local_ip}:389"

    github_user = input_prompt("Enter your GitHub username")
    github_pages_repo = f"{github_user}.github.io"
    cors_origin = f"https://{github_pages_repo}"

    # Create .env.production file
    env_content = f"""LDAP_HOST={ldap_host}
LDAP_BASE_DN=dc=penux,dc=uk
LDAP_ADMIN_DN=cn=admin,dc=penux,dc=uk
LDAP_ADMIN_PASSWORD=admin123
CORS_ORIGIN={cors_origin}
"""

    with open(".env.production", "w") as f:
        f.write(env_content)

    print_success("Environment variables configured")
    print_warning("Please add these to Vercel Dashboard:")
    print(env_content)
    print("\nSteps:")
    print("  1. Go to: https://vercel.com/dashboard")
    print("  2. Select your project")
    print("  3. Settings → Environment Variables")
    print("  4. Add each variable")
    print("  5. Redeploy")

    input_prompt("Press Enter when Vercel environment variables are set")

    print_success("Vercel API configured")
    return True

def phase4_dns_setup(domain: str) -> bool:
    """Phase 4: DNS Configuration"""
    print_header("Phase 4: DNS Configuration")

    github_user = input_prompt("Enter your GitHub username")
    vercel_url = input_prompt("Enter your Vercel API domain (without https://)")
    local_ip = input_prompt("Enter your local machine IP")

    print_info("DNS Records to configure:")
    print(f"\n{Colors.BOLD}Type    Name         Value{Colors.ENDC}")
    print(f"A       @            {local_ip}")
    print(f"A       www          {local_ip}")
    print(f"CNAME   ldap         {github_user}.github.io")
    print(f"CNAME   api.ldap     {vercel_url}")

    print("\nSteps:")
    print("  1. Log in to your domain registrar (Namecheap, GoDaddy, etc.)")
    print("  2. Go to DNS settings for penux.uk")
    print("  3. Add the records above")
    print("  4. Wait 24-48 hours for propagation (usually faster)")

    input_prompt("Press Enter when DNS records are configured")

    print_success("DNS configuration complete")
    return True

def phase5_github_custom_domain(github_user: str, domain: str) -> bool:
    """Phase 5: GitHub Pages Custom Domain"""
    print_header("Phase 5: GitHub Pages Custom Domain")

    custom_domain = f"ldap.{domain}"

    print_warning("Manual steps required:")
    print(f"  1. Go to: https://github.com/{github_user}/{github_user}.github.io/settings/pages")
    print(f"  2. Under 'Custom domain', enter: {custom_domain}")
    print("  3. Click Save")
    print("  4. Check 'Enforce HTTPS'")
    print("  5. Wait for HTTPS certificate (usually ~24 hours)")

    input_prompt("Press Enter when custom domain is configured")

    print_success("GitHub Pages custom domain configured")
    return True

def phase6_connect_ui_to_api(github_user: str, domain: str) -> bool:
    """Phase 6: Connect Web UI to API"""
    print_header("Phase 6: Connect Web UI to API")

    web_url = f"https://ldap.{domain}"
    api_url = f"https://api.ldap.{domain}"

    print_warning("Manual steps required:")
    print(f"  1. Open: {web_url}")
    print("  2. Click Settings (bottom-right)")
    print(f"  3. API Endpoint: {api_url}")
    print("  4. Admin DN: cn=admin,dc=penux,dc=uk")
    print("  5. Admin Password: admin123 (change if modified)")
    print("  6. Click Save")
    print("\n  You should now see:")
    print("    - Connected status")
    print("    - User count")
    print("    - Group count")
    print("    - List of users and groups")

    input_prompt("Press Enter when web UI is connected and working")

    print_success("Web UI connected to API")
    return True

def display_summary(config: Dict) -> None:
    """Display deployment summary"""
    print_header("Deployment Complete!")

    print(f"{Colors.BOLD}Your PenuX LDAP Directory is Now Live!{Colors.ENDC}\n")

    print(f"{Colors.BOLD}Access Points:{Colors.ENDC}")
    print(f"  Local UI:       http://localhost")
    print(f"  LDAP Server:    ldap://localhost:389")
    print(f"  Web Directory:  https://ldap.{config.get('domain', 'penux.uk')}")
    print(f"  API Backend:    https://api.ldap.{config.get('domain', 'penux.uk')}")

    print(f"\n{Colors.BOLD}Credentials:{Colors.ENDC}")
    print(f"  Admin DN:       cn=admin,dc=penux,dc=uk")
    print(f"  Password:       admin123 (CHANGE IN PRODUCTION!)")

    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("  1. Change default passwords")
    print("  2. Set up daily backups")
    print("  3. Create custom users")
    print("  4. Integrate with applications")

    print(f"\n{Colors.BOLD}Management Commands:{Colors.ENDC}")
    print("  cd services/openldap")
    print("  ./manage.ps1 status   (Windows PowerShell)")
    print("  ./manage.sh status    (Linux/macOS)")
    print("  ./manage.ps1 backup   (Create backup)")
    print("  ./manage.ps1 schedule (Daily backups)")

    print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Your enterprise LDAP directory is ready!{Colors.ENDC}\n")

def main():
    """Main automation flow"""
    print_header("PenuX LDAP - Complete Automation")

    print(f"{Colors.BOLD}This script will automate:{Colors.ENDC}")
    print("  1. Local OpenLDAP deployment")
    print("  2. GitHub Pages setup")
    print("  3. Vercel API deployment")
    print("  4. DNS configuration")
    print("  5. Custom domain setup")
    print("  6. Web UI to API connection")

    print(f"\n{Colors.YELLOW}Prerequisites:{Colors.ENDC}")
    print("  ✓ Docker Desktop installed")
    print("  ✓ Git installed")
    print("  ✓ Node.js installed")
    print("  ✓ GitHub account")
    print("  ✓ Vercel account")
    print("  ✓ penux.uk domain registered")
    print("  ✓ Access to domain DNS settings")

    input_prompt("\nPress Enter to begin")

    # Pre-flight checks
    print_header("Pre-flight Checks")

    if not check_docker():
        print_error("Docker check failed. Please install Docker Desktop.")
        return False

    if not check_git():
        print_error("Git check failed. Please install Git.")
        return False

    check_node()

    # Gather configuration
    print_header("Configuration")

    repo_path = input_prompt("Enter penuX repository path", os.getcwd())
    github_user = input_prompt("Enter your GitHub username")
    domain = input_prompt("Enter your domain (default: penux.uk)", "penux.uk")

    config = {
        "repo_path": repo_path,
        "github_user": github_user,
        "domain": domain
    }

    # Execute phases
    phases = [
        ("Local Deployment", lambda: phase1_local_deployment(repo_path)),
        ("GitHub Pages", lambda: phase2_github_pages(repo_path, github_user)),
        ("Vercel API", lambda: phase3_vercel_api(repo_path)),
        ("DNS Setup", lambda: phase4_dns_setup(domain)),
        ("Custom Domain", lambda: phase5_github_custom_domain(github_user, domain)),
        ("Connect UI to API", lambda: phase6_connect_ui_to_api(github_user, domain)),
    ]

    completed_phases = []

    for phase_name, phase_func in phases:
        try:
            if phase_func():
                completed_phases.append(phase_name)
                print_success(f"{phase_name} completed")
            else:
                print_error(f"{phase_name} failed")
                continue_prompt = input_prompt(f"Continue to next phase? (y/n)", "y")
                if continue_prompt.lower() != "y":
                    break
        except KeyboardInterrupt:
            print_warning("\nAutomation interrupted by user")
            break
        except Exception as e:
            print_error(f"Error in {phase_name}: {e}")
            continue_prompt = input_prompt(f"Continue to next phase? (y/n)", "y")
            if continue_prompt.lower() != "y":
                break

    # Display summary
    print_header("Summary")
    print(f"{Colors.BOLD}Completed Phases:{Colors.ENDC}")
    for phase in completed_phases:
        print_success(phase)

    if len(completed_phases) == len(phases):
        display_summary(config)
    else:
        print_warning(f"\nCompleted {len(completed_phases)}/{len(phases)} phases")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Automation cancelled by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.ENDC}")
        sys.exit(1)
