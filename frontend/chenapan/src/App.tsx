import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import type { FileWithPath } from 'react-dropzone';

import axios from 'axios';
import { motion } from 'framer-motion';

// Get the API URL based on environment
const API_URL = process.env.NODE_ENV === 'production'
  ? 'http://neo4j.smalldatabrains.com/api'
  : '/api';

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [legend, setLegend] = useState<{ label: string; color: string }[]>([]);

  const onDrop = useCallback(async (acceptedFiles: FileWithPath[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setImage(URL.createObjectURL(file));
    setMaskImage(null);
    setLegend([]);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post(`${API_URL}/inference`, formData);
      const base64Image = response.data.segmentation_mask;
      const maskUrl = `data:image/png;base64,${base64Image}`;
      setMaskImage(maskUrl);
      setLegend(response.data.legend || []);
    } catch (error) {
      console.error('Error uploading file:', error);
      setMaskImage(null);
      setLegend([]);
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

  useEffect(() => {
  const defaultImage = '/paris.jpg';
  const defaultMask = '/mask_paris.png';

  setImage(defaultImage);
  setMaskImage(defaultMask);

  }, []);


  return (
    <div
      style={{
        width: '100vw',
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(to bottom right, #6366f1, #a855f7)',
        padding: 20,
        boxSizing: 'border-box',
      }}
    >
      <motion.div
        style={{ width: '100%', maxWidth: 1000 }}
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
            <div style={{ marginTop: 24 }}>
              <div
                style={{
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
                    Segmentation Mask
                  </h2>
                  <img
                    src={maskImage}
                    alt="Segmentation result"
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

              {/* Legend Section */}
              {legend.length > 0 && (
                <div style={{ marginTop: 32 }}>
                  <h3
                    style={{
                      fontSize: 16,
                      fontWeight: 600,
                      color: '#374151',
                      marginBottom: 12,
                    }}
                  >
                    Legend
                  </h3>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
                      gap: 12,
                    }}
                  >
                    {legend.map((item, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8,
                          padding: 6,
                          borderRadius: 8,
                          background: '#f3f4f6',
                        }}
                      >
                        <div
                          style={{
                            width: 16,
                            height: 16,
                            borderRadius: 4,
                            backgroundColor: item.color,
                            border: '1px solid #d1d5db',
                          }}
                        />
                        <span style={{ fontSize: 14, color: '#374151' }}>{item.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
