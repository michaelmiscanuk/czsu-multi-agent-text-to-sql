#!/usr/bin/env node

console.log('🔍 Frontend Configuration Debug Info:');
console.log('');

console.log('📋 Environment Variables:');
console.log('  NEXT_PUBLIC_API_BASE:', process.env.NEXT_PUBLIC_API_BASE || '(not set)');
console.log('  NODE_ENV:', process.env.NODE_ENV || '(not set)');
console.log('');

console.log('🌐 Expected Configuration for Local Development:');
console.log('  Frontend URL: http://localhost:3000');
console.log('  Backend URL: http://localhost:8000');
console.log('  API Base should be: http://localhost:8000');
console.log('');

console.log('📝 To fix frontend chat loading issues:');
console.log('  1. Create frontend/.env.local with:');
console.log('     NEXT_PUBLIC_API_BASE=http://localhost:8000');
console.log('  2. Restart the frontend development server');
console.log('  3. Check browser developer console for debug logs');
console.log('');

console.log('🔧 Current working directory:', process.cwd());
console.log('');

console.log('📊 Expected API Endpoints:');
console.log('  - GET /chat-threads (load user threads)');
console.log('  - GET /chat/{thread_id}/messages (load messages)');
console.log('  - POST /analyze (send new messages)');
console.log('  - DELETE /chat/{thread_id} (delete threads)');
console.log('');

console.log('🚀 To test the backend is running:');
console.log('  - Visit: http://localhost:8000/docs');
console.log('  - Should show FastAPI documentation'); 