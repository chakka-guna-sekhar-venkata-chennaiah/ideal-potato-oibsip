
from setuptools import find_packages,setup
from typing import List


hypen_dot='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file=file_path) as f:
        requirements=f.readlines()
        requirements=[i.replace('/n','') for i in requirements]
        if hypen_dot in requirements:
            requirements.remove(hypen_dot)
    return requirements

setup(
name='End to End Machine Learning Industry level Project',
author='Chakka Guna Sekhar Venkata Chennaiah',
author_email='sekharchennaiah12345ctk@gmail.com',
packages=find_packages(),
install_requies=get_requirements('requirements.txt')
)