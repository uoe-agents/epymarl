from setuptools import setup, find_packages

setup(name='myenvs',
      version='0.0.1',
      description='Multi-Agent Simplistic Cooperations Envs',
      #url='https://github.com/openai/multiagent-public',
      author='Arnaud Gardille',
      author_email='arnaud.gardille@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy']
)
