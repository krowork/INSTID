import unittest
import os
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from pipeline_stable_diffusion_xl_instantid import draw_kps

class TestInstantIDPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configuración inicial que se ejecuta una vez para toda la clase"""
        # Crear directorios necesarios
        os.makedirs("test_pipeline", exist_ok=True)
        os.makedirs("test_pipeline/faces", exist_ok=True)
        os.makedirs("test_pipeline/output", exist_ok=True)
        
        # Crear imagen de prueba con un rostro sintético
        cls._create_test_face()

    @classmethod
    def _create_test_face(cls):
        """Crea una imagen de prueba con un rostro sintético simple"""
        img = Image.new('RGB', (512, 512), color='white')
        # Dibujar un rostro esquemático simple
        pixels = np.array(img)
        
        # Óvalo de la cara
        y, x = np.ogrid[:512, :512]
        face_mask = ((x - 256)**2 / (200**2) + (y - 256)**2 / (250**2)) <= 1
        pixels[face_mask] = [255, 220, 180]
        
        # Ojos
        pixels[200:220, 180:220] = [0, 0, 0]  # Ojo izquierdo
        pixels[200:220, 300:340] = [0, 0, 0]  # Ojo derecho
        
        # Nariz
        pixels[250:300, 245:265] = [200, 160, 140]
        
        # Boca
        pixels[350:370, 200:300] = [255, 100, 100]
        
        cls.test_face = Image.fromarray(pixels)
        cls.test_face.save("test_pipeline/faces/test_face.png")

    def test_image_creation(self):
        """Test de creación de imagen"""
        self.assertTrue(os.path.exists("test_pipeline/faces/test_face.png"))
        img = Image.open("test_pipeline/faces/test_face.png")
        self.assertEqual(img.size, (512, 512))
        self.assertEqual(img.mode, 'RGB')

    def test_draw_kps(self):
        """Test de la función de dibujo de keypoints"""
        # Crear keypoints de prueba
        test_kps = np.array([
            [256, 200],  # Ojo izquierdo
            [256, 300],  # Ojo derecho
            [256, 256],  # Nariz
            [200, 350],  # Esquina izquierda boca
            [300, 350]   # Esquina derecha boca
        ])
        
        # Dibujar keypoints
        result = draw_kps(self.test_face, test_kps)
        
        # Verificar que el resultado es una imagen PIL
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_face.size)
        self.assertEqual(result.mode, 'RGB')

    def test_directories(self):
        """Test de la estructura de directorios"""
        self.assertTrue(os.path.exists("test_pipeline"))
        self.assertTrue(os.path.exists("test_pipeline/faces"))
        self.assertTrue(os.path.exists("test_pipeline/output"))

    def test_image_dimensions(self):
        """Test de las dimensiones de la imagen"""
        img = Image.open("test_pipeline/faces/test_face.png")
        self.assertEqual(img.size, (512, 512))
        self.assertEqual(img.mode, 'RGB')
        
        # Convertir a array numpy
        img_array = np.array(img)
        self.assertEqual(img_array.shape, (512, 512, 3))

    @classmethod
    def tearDownClass(cls):
        """Limpieza después de todos los tests"""
        # Eliminar archivos y directorios de prueba
        if os.path.exists("test_pipeline/faces/test_face.png"):
            os.remove("test_pipeline/faces/test_face.png")
        if os.path.exists("test_pipeline/faces"):
            os.rmdir("test_pipeline/faces")
        if os.path.exists("test_pipeline/output"):
            os.rmdir("test_pipeline/output")
        if os.path.exists("test_pipeline"):
            os.rmdir("test_pipeline")

if __name__ == '__main__':
    unittest.main(verbose=True) 