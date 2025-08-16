import json
import os
import sys

class GimnIO:

  def __init__ (self):
    pass
    
  @staticmethod
  def check_and_create_folder(folder_path):
    """
    @brief Checks if a folder exists at the specified path and creates it if it does not exist.
    This function verifies whether the folder at the given path exists. If the folder does not exist,
    it creates the folder and prints a success message. If the folder already exists, it prints a message
    indicating so.
    @param folder_path (str): The path to the folder to be checked or created.
    @return str: The path to the checked or newly created folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"A pasta '{folder_path}' foi criada com sucesso.")
    else:
        print(f"A pasta '{folder_path}' já existe.")
    
    return folder_path

  @staticmethod
  def read_json(file_name):
    """
    @brief Reads a JSON file and returns its contents as a dictionary.

    @param file_name The path to the JSON file to be read.

    @return dict The contents of the JSON file as a Python dictionary.

    @exception FileNotFoundError If the specified file does not exist.
    @exception json.JSONDecodeError If the file is not a valid JSON.

    @note This function assumes the file contains valid JSON data.
    """
    with open(file_name, 'r') as file:
      dic = json.load(file)
    return dic

  @staticmethod
  def save_json(file_name,dict):
        """
        @brief Saves a dictionary to a JSON file.

        This function writes the contents of a dictionary to a file in JSON format.
        It handles values of type str, float, int, list, and numpy.ndarray. Numpy arrays are converted to lists before saving.

        @param file_name The path to the file where the JSON data will be saved.
        @param dict The dictionary containing data to be saved.

        @note This function does not use the built-in json module and may not handle all edge cases of JSON serialization.
        """
        with open(file_name,'w') as file:
                      file.write("{\n")
                      counter = 0
                      items = len(dict)
                      for key,value in dict.items():
                          if isinstance(value,str):
                              file.write(f'\t"{key}":"{value}"')
                          elif isinstance(value,float):
                              file.write(f'\t"{key}":{value}')
                          elif isinstance(value,list):
                              file.write(f'\t"{key}":{value}')
                          elif isinstance(value,np.ndarray):
                              a=value.tolist()
                              file.write(f'\t"{key}":{a}')
                          elif isinstance(value,int):
                              file.write(f'\t"{key}":{value}')
                          if counter<items-1:
                              file.write(",\n")
                          else:
                              file.write("\n")
                          counter+=1
                      file.write("}")
