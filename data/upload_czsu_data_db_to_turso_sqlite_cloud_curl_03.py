# guide - https://turso.tech/blog/migrating-and-importing-sqlite-to-turso-just-got-easier

# use WSL

# example with adjusted TOKENs:


# curl -X POST "https://api.turso.tech/v1/organizations/retko/databases" \
#   -L \
#   -H "Authorization: Bearer eyJhbGciOiJFZERTQSIsInR5cCI6IkpXV2vPz-mpwucoRS0yUhu2tKBZaRDHJ_oEHZxWCsr2PUgLmXIyAVdgBBc3HjM3K23Dw" \
#   -H "Content-Type: application/json" \
#   -d '{
#       "name": "czsudata",
#       "group": "default",
#       "seed": { "type": "database_upload" }
#   }'


# curl -X POST "https://api.turso.tech/v1/organizations/retko/databases/czsudata/auth/tokens" \
#   -L \
#   -H "Authorization: Bearer eyJhbGciOiJFZERTQSIsInR5c.SD5tyNePv3E0h2rx6bqY322vPz-mpwucoRS0yUhu2tKBZaRDHJ_oEHZxWCsr2PUgLmXIyAVdgBBc3HjM3K23Dw"


# {"jwt":"eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE3NjE4NTA5ODcsImlkIjoiOGNjNDUxZGItMDFlYS00NjU2LTllMWQtY2Q5OTdkYWRmNjBlIiwicmlkIjoiOWQyMGYzMDEtNzQ2OS00OWU2LWFkMzEtOTE1MDM3NmY5YzRjIn0.AsswdFiF7Z7_fPGJnN9vCQ9-NxHRoSV5a6XMt27lNzbvt6_Ecy-qsDRl_cnkxJJQgKVtqLpXnddTp38zaEicAg"}


# curl -X POST "https://czsudata-retko.aws-eu-west-1.turso.io/v1/upload" \
#   -H "Authorization: Bearer eyJhbGciOiJFZERTQSIsInMGYzMDEtNzQ2OS00OWU2LWFkMzEtOTE1MDM3NmY5YzRjIn0.AsswdFiF7Z7_fPGJnN9vCQ9-NxHRoSV5a6XMt27lNzbvt6_Ecy-qsDRl_cnkxJJQgKVtqLpXnddTp38zaEicAg" \
#   --data-binary @"/mnt/e/OneDrive/Knowledge Base/0207_GenAI/Code/czsu_home2/czsu-multi-agent-text-to-sql/data/czsu_data.db"
