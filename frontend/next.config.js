/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output: the Docker image ships only .next/standalone + static assets.
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
};

module.exports = nextConfig;
