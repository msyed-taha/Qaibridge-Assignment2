#!/bin/sh
set -eu

psql \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" \
  --set=database="$POSTGRES_DB" \
  --set=app_user="$POSTGRES_APP_USER" \
  --set=app_password="$POSTGRES_APP_PASSWORD" \
  --set=readonly_user="$POSTGRES_READONLY_USER" \
  --set=readonly_password="$POSTGRES_READONLY_PASSWORD" \
  <<'SQL'
CREATE ROLE :"app_user" LOGIN PASSWORD :'app_password';
CREATE ROLE :"readonly_user" LOGIN PASSWORD :'readonly_password';
GRANT CONNECT ON DATABASE :"database" TO :"app_user", :"readonly_user";
GRANT USAGE, CREATE ON SCHEMA public TO :"app_user";
GRANT USAGE ON SCHEMA public TO :"readonly_user";
ALTER DEFAULT PRIVILEGES FOR ROLE :"app_user" IN SCHEMA public
GRANT SELECT ON TABLES TO :"readonly_user";
SQL
