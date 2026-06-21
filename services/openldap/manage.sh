#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_usage() {
    cat << EOF
Usage: ./manage.sh [command]

Commands:
  start         Start OpenLDAP services
  stop          Stop OpenLDAP services
  restart       Restart OpenLDAP services
  status        Show service status
  logs          View service logs (press Ctrl+C to exit)
  logs-ldap     View OpenLDAP logs only
  logs-admin    View phpLDAPadmin logs only
  shell-ldap    Open shell in OpenLDAP container
  shell-admin   Open shell in phpLDAPadmin container
  clean         Stop and remove all containers and volumes (WARNING: data loss!)
  test          Test LDAP connectivity
  add-user      Add a new user to LDAP
  list-users    List all users in LDAP
  help          Show this help message

Examples:
  ./manage.sh start
  ./manage.sh logs
  ./manage.sh test
  ./manage.sh add-user

EOF
}

start_services() {
    echo -e "${BLUE}Starting OpenLDAP services...${NC}"
    docker-compose up -d
    echo -e "${GREEN}Services started!${NC}"
    sleep 2
    docker-compose ps
}

stop_services() {
    echo -e "${BLUE}Stopping OpenLDAP services...${NC}"
    docker-compose down
    echo -e "${GREEN}Services stopped!${NC}"
}

restart_services() {
    echo -e "${BLUE}Restarting OpenLDAP services...${NC}"
    docker-compose restart
    echo -e "${GREEN}Services restarted!${NC}"
}

show_status() {
    echo -e "${BLUE}OpenLDAP Service Status:${NC}"
    docker-compose ps
}

show_logs() {
    docker-compose logs -f
}

test_ldap() {
    echo -e "${BLUE}Testing LDAP connectivity...${NC}"
    source .env 2>/dev/null || source .env.example

    docker run --rm --network openldap_penux_network \
        osixia/openldap:1.5.0 \
        ldapwhoami -H ldap://openldap:389 -D "cn=admin,dc=penux,dc=uk" -w "${LDAP_ADMIN_PASSWORD:-admin123}" && \
    echo -e "${GREEN}✓ LDAP connection successful!${NC}" || \
    echo -e "${RED}✗ LDAP connection failed!${NC}"
}

add_user() {
    read -p "Enter username (uid): " uid
    read -p "Enter full name (cn): " cn
    read -p "Enter email: " email
    read -sp "Enter password: " password
    echo ""

    cat > /tmp/user.ldif << EOF
dn: cn=$cn,ou=users,dc=penux,dc=uk
objectClass: inetOrgPerson
objectClass: organizationalPerson
objectClass: person
cn: $cn
sn: User
givenName: $uid
uid: $uid
mail: $email
userPassword: $password
accountStatus: active
description: Added via script
EOF

    source .env 2>/dev/null || source .env.example

    docker exec penux-openldap ldapadd -x -D "cn=admin,dc=penux,dc=uk" -w "${LDAP_ADMIN_PASSWORD:-admin123}" -f /tmp/user.ldif

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ User $uid added successfully!${NC}"
    else
        echo -e "${RED}✗ Failed to add user!${NC}"
    fi

    rm -f /tmp/user.ldif
}

list_users() {
    echo -e "${BLUE}Listing all users in LDAP:${NC}"
    source .env 2>/dev/null || source .env.example

    docker exec penux-openldap ldapsearch -x -D "cn=admin,dc=penux,dc=uk" -w "${LDAP_ADMIN_PASSWORD:-admin123}" \
        -b "ou=users,dc=penux,dc=uk" "(objectClass=inetOrgPerson)" uid mail cn
}

case "${1:-help}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    logs-ldap)
        docker-compose logs -f openldap
        ;;
    logs-admin)
        docker-compose logs -f phpldapadmin
        ;;
    shell-ldap)
        docker exec -it penux-openldap /bin/bash
        ;;
    shell-admin)
        docker exec -it penux-phpldapadmin /bin/bash
        ;;
    test)
        test_ldap
        ;;
    add-user)
        add_user
        ;;
    list-users)
        list_users
        ;;
    clean)
        read -p "WARNING: This will delete all data! Continue? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            docker-compose down -v
            echo -e "${GREEN}Cleaned!${NC}"
        else
            echo "Cancelled"
        fi
        ;;
    help)
        show_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_usage
        exit 1
        ;;
esac
