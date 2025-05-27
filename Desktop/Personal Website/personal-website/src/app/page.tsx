import React from 'react';
import { EnvelopeIcon, PhoneIcon, GlobeAltIcon } from '@heroicons/react/24/outline';
import { AnimatedSection } from '@/components/AnimatedSection';
import CosmicBackground from '@/components/CosmicBackground';
import PhotoGallery from '@/components/PhotoGallery';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <CosmicBackground />
      {/* Hero Section */}
      <section className="min-h-screen flex items-center justify-center relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-blue-100/20 via-transparent to-transparent dark:from-blue-900/20"></div>
        <div className="container mx-auto px-4 py-16 text-center relative z-10">
          <AnimatedSection>
            <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 hover:scale-105 transition-transform duration-300">
              Hansheng Zhu
            </h1>
          </AnimatedSection>
          <AnimatedSection delay={0.2}>
            <p className="text-xl text-gray-600 dark:text-gray-300 mb-8 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300">
              AI Engineer & Researcher at UPenn
            </p>
          </AnimatedSection>
          <AnimatedSection delay={0.4}>
            <div className="flex justify-center space-x-6">
              <a href="mailto:hanszhu@seas.upenn.edu" className="flex items-center space-x-2 hover:text-blue-600 transition-all duration-300 hover:scale-110">
                <EnvelopeIcon className="h-5 w-5" />
                <span>hanszhu@seas.upenn.edu</span>
              </a>
              <a href="tel:+14458005280" className="flex items-center space-x-2 hover:text-blue-600 transition-all duration-300 hover:scale-110">
                <PhoneIcon className="h-5 w-5" />
                <span>(445) 800-5280</span>
              </a>
              <a href="https://github.com/hanshengzhu0001" target="_blank" rel="noopener noreferrer" className="flex items-center space-x-2 hover:text-blue-600 transition-all duration-300 hover:scale-110">
                <GlobeAltIcon className="h-5 w-5" />
                <span>GitHub</span>
              </a>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Education Section */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/50 to-transparent dark:via-gray-900/50"></div>
        <div className="container mx-auto px-4 relative z-10">
          <h2 className="text-3xl font-bold mb-8 text-center hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300">Education</h2>
          <div className="max-w-3xl mx-auto">
            <AnimatedSection>
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-gray-100 dark:border-gray-700 hover:shadow-xl hover:scale-[1.02] transition-all duration-300">
                <h3 className="text-xl font-semibold mb-2">University of Pennsylvania</h3>
                <p className="text-gray-600 dark:text-gray-300 mb-2">School of Engineering and Applied Sciences</p>
                <p className="text-gray-600 dark:text-gray-300 mb-4">Rising Junior</p>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  Candidate for Bachelor of Engineering in Artificial Intelligence
                </p>
                <div className="mt-4">
                  <h4 className="font-semibold mb-2">Relevant Coursework:</h4>
                  <p className="text-gray-600 dark:text-gray-300">
                    Machine Learning, Signal Processing, Data Structures and Algorithms, Stochastic Processes, 
                    Artificial Intelligence, Machine Perception, Neural Networks, Brain Computer Interface, OOP
                  </p>
                </div>
              </div>
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* Fun Facts Section */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-gray-50/50 to-transparent dark:via-gray-800/50"></div>
        <div className="container mx-auto px-4 relative z-10">
          <h2 className="text-3xl font-bold mb-8 text-center hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300">Fun Facts About Me</h2>
          <div className="max-w-3xl mx-auto">
            <AnimatedSection>
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-gray-100 dark:border-gray-700 hover:shadow-xl hover:scale-[1.02] transition-all duration-300">
                <ul className="space-y-4 text-gray-600 dark:text-gray-300">
                  <li className="flex items-start group">
                    <span className="text-blue-600 dark:text-blue-400 mr-2 group-hover:scale-125 transition-transform duration-300">•</span>
                    <span className="group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-300">I love playing tennis but not so good</span>
                  </li>
                  <li className="flex items-start group">
                    <span className="text-blue-600 dark:text-blue-400 mr-2 group-hover:scale-125 transition-transform duration-300">•</span>
                    <span className="group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-300">Playing in a 3v3 basketball tournament this summer</span>
                  </li>
                  <li className="flex items-start group">
                    <span className="text-blue-600 dark:text-blue-400 mr-2 group-hover:scale-125 transition-transform duration-300">•</span>
                    <span className="group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-300">I love singing in the shower (my roommates might disagree)</span>
                  </li>
                  <li className="flex items-start group">
                    <span className="text-blue-600 dark:text-blue-400 mr-2 group-hover:scale-125 transition-transform duration-300">•</span>
                    <span className="group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-300">Certified bronze level ballroom dancer</span>
                  </li>
                  <li className="flex items-start group">
                    <span className="text-blue-600 dark:text-blue-400 mr-2 group-hover:scale-125 transition-transform duration-300">•</span>
                    <span className="group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-300">I sing bass in an acappella group (Penn Enchord)</span>
                  </li>
                  <li className="flex items-start group">
                    <span className="text-blue-600 dark:text-blue-400 mr-2 group-hover:scale-125 transition-transform duration-300">•</span>
                    <span className="group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-300">I play clarinet and piano</span>
                  </li>
                </ul>
              </div>
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* Experience Section */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/50 to-transparent dark:via-gray-900/50"></div>
        <div className="container mx-auto px-4 relative z-10">
          <h2 className="text-3xl font-bold mb-8 text-center hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300">Work Experience</h2>
          <div className="max-w-3xl mx-auto space-y-6">
            <AnimatedSection>
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-gray-100 dark:border-gray-700 hover:shadow-xl hover:scale-[1.02] transition-all duration-300">
                <h3 className="text-xl font-semibold mb-2">Software Development Intern</h3>
                <p className="text-gray-600 dark:text-gray-300 mb-2">Astoria AI • Jan 2025 - Present</p>
                <ul className="list-disc list-inside text-gray-600 dark:text-gray-300">
                  <li>Designed and optimized LLM-as-a-Judge evaluation pipeline for conversational AI</li>
                  <li>Built scalable frontend modules for talent management platform</li>
                </ul>
              </div>
            </AnimatedSection>
            <AnimatedSection delay={0.2}>
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-gray-100 dark:border-gray-700 hover:shadow-xl hover:scale-[1.02] transition-all duration-300">
                <h3 className="text-xl font-semibold mb-2">Deep Learning Research Intern</h3>
                <p className="text-gray-600 dark:text-gray-300 mb-2">Thomas Jefferson University Hospital • Oct 2024 - Present</p>
                <ul className="list-disc list-inside text-gray-600 dark:text-gray-300">
                  <li>Performed image segmentation on digital subtraction angiograms through neural networks</li>
                  <li>Implemented vision transformer to predict risk of capillary abnormalities</li>
                </ul>
              </div>
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* Projects Section */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-gray-50/50 to-transparent dark:via-gray-800/50"></div>
        <div className="container mx-auto px-4 relative z-10">
          <h2 className="text-3xl font-bold mb-8 text-center hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300">Technical Projects</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            <AnimatedSection>
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-gray-100 dark:border-gray-700 hover:shadow-xl hover:scale-[1.02] transition-all duration-300">
                <h3 className="text-xl font-semibold mb-2">Alziaid iOS App</h3>
                <p className="text-gray-600 dark:text-gray-300 mb-4">Swift, React</p>
                <p className="text-gray-600 dark:text-gray-300">
                  Led team of 4 to build a Alzheimer's diagnosis app with real-time face movement analysis; deployed for 100+ beta users.
                </p>
              </div>
            </AnimatedSection>
            <AnimatedSection delay={0.2}>
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-gray-100 dark:border-gray-700 hover:shadow-xl hover:scale-[1.02] transition-all duration-300">
                <h3 className="text-xl font-semibold mb-2">Plant Disease Detection</h3>
                <p className="text-gray-600 dark:text-gray-300 mb-4">PyTorch, ResNet</p>
                <p className="text-gray-600 dark:text-gray-300">
                  Trained CNN model for leaf image classification (95% accuracy); can be integrated into open-source agriculture toolkit.
                </p>
              </div>
            </AnimatedSection>
            <AnimatedSection delay={0.3}>
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-gray-100 dark:border-gray-700 hover:shadow-xl hover:scale-[1.02] transition-all duration-300">
                <h3 className="text-xl font-semibold mb-2">MedScanner</h3>
                <p className="text-gray-600 dark:text-gray-300 mb-4">React, AWS, BioBERT</p>
                <p className="text-gray-600 dark:text-gray-300">
                  Full-stack app for drug dosage search and assistance; deployed with and optimized AWS Lambda functions to handle 500+ daily queries.
                </p>
              </div>
            </AnimatedSection>
            <AnimatedSection delay={0.4}>
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-gray-100 dark:border-gray-700 hover:shadow-xl hover:scale-[1.02] transition-all duration-300">
                <h3 className="text-xl font-semibold mb-2">Finger Movement Prediction</h3>
                <p className="text-gray-600 dark:text-gray-300 mb-4">Python, CuML, scikit-learn</p>
                <p className="text-gray-600 dark:text-gray-300">
                  Developed a multiscale high-γ/β feature extraction pipeline with GPU-accelerated Random Forests.
                </p>
              </div>
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* Add Photo Gallery before the Current Research section */}
      <PhotoGallery />

      {/* Current Research Section */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/50 to-transparent dark:via-gray-900/50"></div>
        <div className="container mx-auto px-4 relative z-10">
          <h2 className="text-3xl font-bold mb-8 text-center hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300">Current Research</h2>
          <div className="max-w-3xl mx-auto">
            <AnimatedSection>
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-gray-100 dark:border-gray-700 hover:shadow-xl hover:scale-[1.02] transition-all duration-300">
                <p className="text-gray-600 dark:text-gray-300">
                  Working with Professor Chris Callison-Burch on extending MOLMO - a family of open state-of-the-art multimodal AI models - which applies dense audio captioning and combining ViT-based image encoding with LLM-based decoding.
                </p>
              </div>
            </AnimatedSection>
          </div>
        </div>
      </section>
    </main>
  );
}
