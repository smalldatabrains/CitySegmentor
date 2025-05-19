import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import type { FileWithPath } from 'react-dropzone';

import axios from 'axios';
import { motion } from 'framer-motion';

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [maskImage, setMaskImage] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: FileWithPath[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setImage(URL.createObjectURL(file));
    setMaskImage(null); // reset mask while loading new

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/api/inference', formData);
      const maskUrl = URL.createObjectURL(response.data);
      setMaskImage(maskUrl);
    } catch (error) {
      console.error('Error uploading file:', error);
      setMaskImage(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    maxFiles: 1,
    multiple: undefined,
    onDragEnter: undefined,
    onDragOver: undefined,
    onDragLeave: undefined
  });

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(to bottom right, #6366f1, #a855f7)',
        padding: 20,
        boxSizing: 'border-box',
      }}
    >
      <motion.div
        style={{ width: '100%', maxWidth: 960 }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div
          style={{
            background: 'white',
            borderRadius: 16,
            boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
            padding: 24,
            boxSizing: 'border-box',
          }}
        >
          <div
            {...getRootProps()}
            style={{
              border: '4px dashed',
              borderColor: isDragActive ? '#6366f1' : '#d1d5db',
              backgroundColor: isDragActive ? '#e0e7ff' : '#f9fafb',
              borderRadius: 16,
              padding: 40,
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.3s ease-in-out',
            }}
          >
            <input {...getInputProps()} />
            {isDragActive ? (
              <p style={{ fontSize: 18, color: '#4f46e5' }}>Drop the image here...</p>
            ) : (
              <p style={{ fontSize: 18, color: '#374151' }}>
                Drag & drop an image here, or click to select one
              </p>
            )}
          </div>

          {image && maskImage && (
            <div
              style={{
                marginTop: 24,
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 24,
              }}
            >
              <div style={{ textAlign: 'center' }}>
                <h2
                  style={{
                    marginBottom: 8,
                    fontSize: 16,
                    fontWeight: 600,
                    color: '#374151',
                  }}
                >
                  Original Image
                </h2>
                <img
                  src={image}
                  alt="Uploaded preview"
                  style={{
                    maxHeight: 384,
                    margin: '0 auto',
                    borderRadius: 16,
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                    maxWidth: '100%',
                    objectFit: 'contain',
                  }}
                />
              </div>
              <div style={{ textAlign: 'center' }}>
                <h2
                  style={{
                    marginBottom: 8,
                    fontSize: 16,
                    fontWeight: 600,
                    color: '#374151',
                  }}
                >
                  UNet Mask
                </h2>
                <img
                  src={maskImage}
                  alt="UNet result"
                  style={{
                    maxHeight: 384,
                    margin: '0 auto',
                    borderRadius: 16,
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                    maxWidth: '100%',
                    objectFit: 'contain',
                  }}
                />
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
