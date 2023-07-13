const fs = require('fs');

const secureConfig = JSON.parse(fs.readFileSync('secure-config.json'));

const envVars = Object.entries(secureConfig)
	.map(([key, value]) => `${key}=${value}`)
	.join('\n');

fs.writeFileSync('.env', envVars);
