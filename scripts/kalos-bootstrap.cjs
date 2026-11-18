const fs = require('fs');
const crypto = require('crypto');
const directory = process.argv[2] || '/deploy';
const own = path => { if (process.getuid() === 0) fs.chownSync(path, 10001, 10001); };

let input = '';
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  const effective = Object.fromEntries(input.split('\n').filter(line => line.includes('=')).map(line => {
    const separator = line.indexOf('=');
    return [line.slice(0, separator), line.slice(separator + 1)];
  }));
  const path = `${directory}/.env`;
  const original = fs.readFileSync(path, 'utf8');
  const present = new Set([...original.matchAll(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=/gm)].map(match => match[1]));
  const secret = () => crypto.randomBytes(32).toString('hex');
  const token = effective.KALOS_ENGINE_TOKEN || secret();
  const privatePath = `${directory}/secrets/engine-private.pem`;
  const publicPath = `${directory}/secrets/engine-public.jwks`;
  const existingKid = fs.existsSync(publicPath) ? JSON.parse(fs.readFileSync(publicPath, 'utf8')).keys[0]?.kid : undefined;
  const kid = effective.KALOS_ENGINE_JWT_KID || existingKid || `local-${crypto.randomBytes(8).toString('hex')}`;
  fs.mkdirSync(`${directory}/secrets`, { recursive: true });
  if (!fs.existsSync(privatePath)) {
    if (fs.existsSync(publicPath)) throw new Error('Private key missing but public key exists; restore the matching private key from backup.');
    const { privateKey } = crypto.generateKeyPairSync('ed25519');
    fs.writeFileSync(privatePath, privateKey.export({ type: 'pkcs8', format: 'pem' }), { flag: 'wx', mode: 0o400 });
    own(privatePath);
  }
  if (!fs.existsSync(publicPath)) {
    const publicKey = crypto.createPublicKey(fs.readFileSync(privatePath));
    fs.writeFileSync(publicPath, JSON.stringify({ keys: [{ ...publicKey.export({ format: 'jwk' }), kid, use: 'sig', alg: 'EdDSA' }] }), { flag: 'wx' });
  }
  const tenantsPath = `${directory}/secrets/tenants.json`;
  if (!fs.existsSync(tenantsPath)) {
    fs.writeFileSync(tenantsPath, '{}', { flag: 'wx', mode: 0o600 });
    own(tenantsPath);
  }
  const defaults = {
    KALOS_ENGINE_TOKEN: token,
    KALOS_AUTH_TOKENS: JSON.stringify([{ token_sha256: crypto.createHash('sha256').update(token).digest('hex'), subject: 'local-ops', tenant: 'local', scopes: ['read', 'write'], service_account: true }]),
    KALOS_ANON_SALT: secret(), BETTER_AUTH_SECRET: secret(),
    BETTER_AUTH_URL: 'http://localhost:3000', KALOS_PUBLIC_URL: 'http://localhost:3000',
    KALOS_CORS_ORIGINS: 'http://localhost:3000', KALOS_ENGINE_JWT_KID: kid,
    GOOGLE_CLIENT_ID: '', GOOGLE_CLIENT_SECRET: '',
    AUTH_TRUSTED_PROXIES: 'none', AUTH_ACCEPT_NO_PROXY: '1',
    KALOS_TORCH_THREADS: '0', KALOS_ENGINE_CPUS: '0'
  };
  const additions = Object.entries(defaults).filter(([key]) => !present.has(key));
  if (additions.length) fs.appendFileSync(path, (original && !original.endsWith('\n') ? '\n' : '') + additions.map(([key, value]) => `${key}='${value}'`).join('\n') + '\n');
});
