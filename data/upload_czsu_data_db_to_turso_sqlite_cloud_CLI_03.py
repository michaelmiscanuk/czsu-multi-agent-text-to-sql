# sqlitecloud.io
# manually via UI upload to cloud sqlite db
# https://dashboard.sqlitecloud.io/
# go to your project, choose "Upload Database" and drag czsu_data.db file there

# On Turso.com
# https://app.turso.tech/retko/databases/czsudata/data
# migrate local to cloud db: https://docs.turso.tech/cloud/migrate-to-turso

# Install sqlite3  tools - https://sqlite.org/2025/sqlite-tools-win-x64-3500400.zip
# SETUP into PATH environment variable

# cd data
# sqlite3 czsu_data.db

# PRAGMA journal_mode='wal';
# PRAGMA wal_checkpoint(truncate);
# PRAGMA journal_mode;
# .exit


# Install WSL and Ubuntu Linux on Windows
# wsl --uninstall
# wsl --install


# Install Turso CLI - https://docs.turso.tech/cli/installation
# curl -sSfL https://get.tur.so/install.sh | bash

# verify installation - in new shell session
# turso

# turso auth signup
# or
# turso auth signup --headless
# turso db import "/mnt/e/OneDrive/Knowledge Base/0207_GenAI/Code/czsu_home2/czsu-multi-agent-text-to-sql/data/czsu_data.db"
