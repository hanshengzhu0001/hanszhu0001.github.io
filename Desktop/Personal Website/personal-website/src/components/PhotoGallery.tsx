'use client';

import React, { useState } from 'react';
import { AnimatedSection } from './AnimatedSection';

interface Photo {
  src: string;
  alt: string;
  description: string;
}

const PhotoGallery = () => {
  const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null);

  // Add your photos here
  const photos: Photo[] = [
    {
      src: '/images/food-table.jpg',
      alt: 'Homemade food on a table',
      description: ''
    },
    {
      src: '/images/kirby-poker.jpg',
      alt: 'Poker chips with Kirby figure',
      description: ''
    },
    {
      src: '/images/friends-flowers.jpg',
      alt: 'Group of friends with flowers',
      description: ''
    },
    {
      src: '/images/pork-belly.jpg',
      alt: 'Pork belly cooking in a pan',
      description: ''
    },
    {
      src: '/images/alcaraz-tennis.jpg',
      alt: 'Carlos Alcaraz serving in tennis',
      description: ''
    },
    {
      src: '/images/hotpot.jpg',
      alt: 'Hotpot with friends',
      description: ''
    },
  ];

  return (
    <section className="py-20 relative">
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-gray-50/50 to-transparent dark:via-gray-800/50"></div>
      <div className="container mx-auto px-4 relative z-10">
        <h2 className="text-3xl font-bold mb-8 text-center hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300">
          Photo Gallery
        </h2>
        <div className="max-w-6xl mx-auto">
          <AnimatedSection>
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-gray-100 dark:border-gray-700">
              {photos.length === 0 ? (
                <div className="text-center py-12">
                  <p className="text-gray-600 dark:text-gray-300 mb-4">
                    Add your photos to the gallery by updating the photos array in PhotoGallery.tsx
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Place your images in the public/images directory
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {photos.map((photo, index) => (
                    <div
                      key={index}
                      className="relative group cursor-pointer overflow-hidden rounded-lg"
                      onClick={() => setSelectedPhoto(photo)}
                    >
                      <img
                        src={photo.src}
                        alt={photo.alt}
                        className="w-full h-64 object-cover transform group-hover:scale-110 transition-transform duration-300"
                      />
                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all duration-300 flex items-center justify-center">
                        <p className="text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300 text-center p-4">
                          {photo.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </AnimatedSection>
        </div>
      </div>

      {/* Modal for enlarged photo view */}
      {selectedPhoto && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedPhoto(null)}
        >
          <div className="relative max-w-4xl w-full">
            <img
              src={selectedPhoto.src}
              alt={selectedPhoto.alt}
              className="w-full h-auto rounded-lg"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-4 rounded-b-lg">
              <p className="text-lg font-semibold">{selectedPhoto.alt}</p>
              <p className="text-sm mt-2">{selectedPhoto.description}</p>
            </div>
            <button
              className="absolute top-4 right-4 text-white hover:text-gray-300 transition-colors duration-300"
              onClick={() => setSelectedPhoto(null)}
            >
              <svg
                className="w-8 h-8"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>
      )}
    </section>
  );
};

export default PhotoGallery; 