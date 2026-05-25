/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    // Higgsfield + Cloudinary serve generated landscape results.
    remotePatterns: [
      { protocol: "https", hostname: "*.higgsfield.ai" },
      { protocol: "https", hostname: "res.cloudinary.com" },
    ],
  },
};
export default nextConfig;
