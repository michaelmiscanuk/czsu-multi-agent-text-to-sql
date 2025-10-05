# 🔐 Environment Variables Setup Guide

## Overview

This guide explains how to securely configure environment variables for your CZSU Multi-Agent Text-to-SQL API deployment on Railway.

## 🛡️ Security Approach

- ✅ **`railway.toml`** - Safe to commit (no secrets)
- ✅ **Railway Dashboard** - Store actual secret values
- ✅ **Variable References** - Use `${{Railway.VARIABLE_NAME}}` syntax
- ❌ **Never commit** actual API keys or database URLs

## 🚀 Railway Setup Steps

### 1. Deploy with Current Configuration
First, deploy with the current `railway.toml` (no secrets):
```bash
git add railway.toml Dockerfile .dockerignore
git commit -m "Add Railway Docker configuration"
git push
```

### 2. Add Environment Variables in Railway Dashboard

Go to your Railway project → Service → **Variables** tab and add:

#### 🤖 AI/ML API Keys
```
OPENAI_API_KEY = sk-your-actual-openai-key-here
LANGSMITH_API_KEY = your-langsmith-key-here
COHERE_API_KEY = your-cohere-key-here
ANTHROPIC_API_KEY = your-anthropic-key-here
LLAMA_PARSE_API_KEY = your-llamaparse-key-here
```

#### 🗄️ Database URLs
```
DATABASE_URL = postgresql://user:pass@host:port/dbname
POSTGRES_URL = postgresql://user:pass@host:port/dbname
```

#### 🔑 Authentication Secrets
```
JWT_SECRET_KEY = your-jwt-secret-key-here
AUTH0_CLIENT_ID = your-auth0-client-id
AUTH0_CLIENT_SECRET = your-auth0-client-secret
AUTH0_DOMAIN = your-domain.auth0.com
```

#### 🔧 Optional Services
```
REDIS_URL = redis://user:pass@host:port
WEBHOOK_SECRET = your-webhook-secret
```

### 3. Enable Variable References (Optional)
If you want to reference Railway-managed variables in `railway.toml`, uncomment these lines:
```toml
OPENAI_API_KEY = "${{Railway.OPENAI_API_KEY}}"
DATABASE_URL = "${{Railway.DATABASE_URL}}"
```

## 🔄 Deployment Workflow

### Option 1: Railway Dashboard Only (Recommended)
1. Set all variables in Railway dashboard
2. Keep `railway.toml` without secrets
3. Deploy - Railway injects variables automatically

### Option 2: Mixed Approach
1. Set secrets in Railway dashboard
2. Reference them in `railway.toml` with `${{Railway.VARIABLE_NAME}}`
3. Deploy with updated configuration

## 🧪 Testing Environment Variables

### Check if variables are loaded:
```bash
# Railway CLI (if installed)
railway run python -c "import os; print('OpenAI key loaded:', bool(os.getenv('OPENAI_API_KEY')))"

# Or check in deployed app logs
# Variables should appear as: OPENAI_API_KEY=sk-****(redacted)
```

### Debug environment in container:
```bash
# View non-secret environment variables
railway logs

# Check specific variable exists (without revealing value)
railway run python -c "import os; print([k for k in os.environ.keys() if 'OPENAI' in k])"
```

## 📋 Environment Variables Checklist

Copy this checklist and check off as you add each variable in Railway:

### Required for Basic Functionality:
- [ ] `OPENAI_API_KEY` - OpenAI API access
- [ ] `DATABASE_URL` - PostgreSQL database connection

### Required for Full Functionality:
- [ ] `LANGSMITH_API_KEY` - LangSmith tracing
- [ ] `COHERE_API_KEY` - Cohere reranking
- [ ] `LLAMA_PARSE_API_KEY` - PDF parsing

### Authentication (if using):
- [ ] `JWT_SECRET_KEY` - JWT token signing
- [ ] `AUTH0_CLIENT_ID` - Auth0 client ID
- [ ] `AUTH0_CLIENT_SECRET` - Auth0 client secret
- [ ] `AUTH0_DOMAIN` - Auth0 domain

### Optional:
- [ ] `ANTHROPIC_API_KEY` - Claude API access
- [ ] `REDIS_URL` - Redis caching
- [ ] `WEBHOOK_SECRET` - Webhook verification

## 🔍 Common Issues & Solutions

### Issue: "Missing API key" errors
**Solution**: Verify variable names match exactly in Railway dashboard

### Issue: Variables not loading
**Solution**: 
1. Check Railway deployment logs
2. Ensure variable names have no typos
3. Redeploy after adding variables

### Issue: Authentication errors
**Solution**: 
1. Verify database URL format: `postgresql://user:pass@host:port/db`
2. Check JWT secret is properly set
3. Validate Auth0 configuration

## 🔐 Security Best Practices

1. **Rotate Keys Regularly**: Update API keys every 3-6 months
2. **Monitor Usage**: Check Railway logs for suspicious activity
3. **Limit Scope**: Use API keys with minimal required permissions
4. **Use Strong Secrets**: Generate JWT secrets with high entropy
5. **Environment Separation**: Different keys for development/production

## 📊 Railway Variable Management

### View Variables:
```bash
railway variables
```

### Set Variable via CLI:
```bash
railway variables set OPENAI_API_KEY=sk-your-key-here
```

### Delete Variable:
```bash
railway variables delete VARIABLE_NAME
```

This approach ensures your secrets are secure while maintaining a clean, version-controlled configuration!