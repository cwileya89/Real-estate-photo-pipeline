import { v2 as cloudinary } from "cloudinary";

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure: true,
});

export default cloudinary;

export async function uploadDataUrl(dataUrl: string, folder: string): Promise<string> {
  const result = await cloudinary.uploader.upload(dataUrl, { folder });
  return result.secure_url;
}

export async function uploadFromUrl(url: string, folder: string): Promise<string> {
  const result = await cloudinary.uploader.upload(url, { folder });
  return result.secure_url;
}
