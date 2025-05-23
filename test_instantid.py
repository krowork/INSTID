import unittest
import os
from PIL import Image
import numpy as np
import json
import shutil

class TestInstantIDBasic(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para cada test"""
        # Crear directorios necesarios para tests
        os.makedirs("test_images", exist_ok=True)
        os.makedirs("test_checkpoints", exist_ok=True)
        
    def test_directories_creation(self):
        """Test de creación de directorios"""
        required_dirs = ["test_images", "test_checkpoints"]
        for dir_name in required_dirs:
            self.assertTrue(os.path.exists(dir_name), f"El directorio {dir_name} no existe")
            
    def test_nested_directories_creation(self):
        """Test de creación de directorios anidados"""
        nested_path = "test_checkpoints/ControlNetModel/subfolder"
        os.makedirs(nested_path, exist_ok=True)
        self.assertTrue(os.path.exists(nested_path))
        self.assertTrue(os.path.isdir(nested_path))
    
    def test_image_creation(self):
        """Test de creación y manipulación de imágenes"""
        # Crear una imagen de prueba
        test_image = Image.new('RGB', (640, 640), color='white')
        
        # Verificar dimensiones
        self.assertEqual(test_image.size, (640, 640))
        self.assertEqual(test_image.mode, 'RGB')
        
        # Guardar y cargar la imagen
        test_path = "test_images/test.png"
        test_image.save(test_path)
        
        # Usar with para manejar el archivo correctamente
        with Image.open(test_path) as loaded_image:
            # Verificar que la imagen se guardó y cargó correctamente
            self.assertTrue(os.path.exists(test_path))
            self.assertEqual(loaded_image.size, test_image.size)
            self.assertEqual(loaded_image.mode, test_image.mode)
    
    def test_image_formats(self):
        """Test de diferentes formatos de imagen"""
        test_image = Image.new('RGB', (640, 640), color='white')
        formats = {
            'PNG': 'test_images/test.png',
            'JPEG': 'test_images/test.jpg',
            'BMP': 'test_images/test.bmp'
        }
        
        for fmt, path in formats.items():
            test_image.save(path)
            with Image.open(path) as img:
                self.assertTrue(os.path.exists(path))
                self.assertEqual(img.size, (640, 640))
    
    def test_numpy_conversion(self):
        """Test de conversión entre PIL y numpy arrays"""
        # Crear una imagen de prueba
        test_image = Image.new('RGB', (640, 640), color='white')
        
        # Convertir a numpy array
        img_array = np.array(test_image)
        
        # Verificar forma y tipo del array
        self.assertEqual(img_array.shape, (640, 640, 3))
        self.assertEqual(img_array.dtype, np.uint8)
        
        # Convertir de vuelta a PIL
        pil_image = Image.fromarray(img_array)
        
        # Verificar que la conversión fue correcta
        self.assertEqual(pil_image.size, test_image.size)
        self.assertEqual(pil_image.mode, test_image.mode)
    
    def test_numpy_array_operations(self):
        """Test de operaciones con arrays numpy"""
        # Crear array de prueba
        test_array = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Llenar con valores de prueba
        test_array[100:200, 100:200] = [255, 0, 0]  # Cuadrado rojo
        
        # Verificar valores
        self.assertTrue(np.all(test_array[100:200, 100:200, 0] == 255))
        self.assertTrue(np.all(test_array[100:200, 100:200, 1:] == 0))
        
        # Convertir a imagen
        img = Image.fromarray(test_array)
        self.assertEqual(img.size, (640, 640))
    
    def test_file_paths(self):
        """Test de rutas de archivos"""
        test_paths = [
            "checkpoints/ControlNetModel/config.json",
            "checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors",
            "checkpoints/ip-adapter.bin"
        ]
        
        # Verificar que las rutas son válidas
        for path in test_paths:
            self.assertIsInstance(path, str)
            self.assertTrue(len(path) > 0)
            self.assertTrue('/' in path)
    
    def test_json_config(self):
        """Test de manejo de archivos de configuración JSON"""
        config = {
            "model_path": "checkpoints/model.safetensors",
            "device": "cuda",
            "parameters": {
                "steps": 30,
                "guidance_scale": 7.5,
                "strength": 0.8
            }
        }
        
        # Guardar configuración
        config_path = "test_checkpoints/config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Cargar y verificar configuración
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(config, loaded_config)
        self.assertEqual(loaded_config["parameters"]["steps"], 30)
    
    def test_image_dimensions(self):
        """Test de diferentes dimensiones de imagen"""
        dimensions = [(640, 640), (512, 512), (1024, 1024)]
        
        for width, height in dimensions:
            img = Image.new('RGB', (width, height), color='white')
            self.assertEqual(img.size, (width, height))
            
            # Convertir a numpy y verificar
            arr = np.array(img)
            self.assertEqual(arr.shape, (height, width, 3))
    
    def test_error_handling(self):
        """Test de manejo de errores"""
        # Intentar abrir un archivo que no existe
        with self.assertRaises(FileNotFoundError):
            Image.open("archivo_inexistente.png")
        
        # Intentar crear un directorio en una ruta inválida
        with self.assertRaises(OSError):
            os.makedirs("/ruta/invalida/test")
            
        # Intentar guardar en un formato no soportado
        img = Image.new('RGB', (640, 640))
        with self.assertRaises(ValueError):
            img.save("test.xyz")
    
    def test_file_operations(self):
        """Test de operaciones con archivos"""
        # Crear archivo de prueba
        test_file = "test_checkpoints/test.txt"
        content = "Contenido de prueba"
        
        # Escribir
        with open(test_file, 'w') as f:
            f.write(content)
        
        # Verificar existencia
        self.assertTrue(os.path.exists(test_file))
        
        # Leer y verificar contenido
        with open(test_file, 'r') as f:
            read_content = f.read()
        self.assertEqual(content, read_content)
        
        # Eliminar
        os.remove(test_file)
        self.assertFalse(os.path.exists(test_file))
    
    def tearDown(self):
        """Limpieza después de cada test"""
        # Eliminar archivos de prueba
        test_files = [
            "test_images/test.png",
            "test_images/test.jpg",
            "test_images/test.bmp",
            "test_checkpoints/config.json"
        ]
        
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
        
        # Eliminar directorios de prueba
        test_dirs = ["test_images", "test_checkpoints"]
        for dir_name in test_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)

if __name__ == '__main__':
    unittest.main(verbose=True) 