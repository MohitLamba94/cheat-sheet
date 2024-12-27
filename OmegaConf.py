### A Human Readable way of creating Dictionaries and configs for ML training

from omegaconf import OmegaConf

yaml_str = """
app:
  name: MyApp
  version: 1.0
  environment: production
  features:
    - name: feature1
      enabled: true
    - name: feature2
      enabled: false
database:
  primary:
    host: localhost
    port: 5432
    credentials:
      user: admin
      password: secret
  secondary:
    host: localhost
    port: 3306
    credentials:
      user: backup
      password: backup_secret
"""

config = OmegaConf.create(yaml_str)

print(config.app.name)                   # Output: MyApp
print(config.app.features[0].name)       # Output: feature1
print(config.database.primary.host)      # Output: localhost
print(config.database.secondary.credentials.user)  # Output: backup
print(config) # prints like a dictionary!
'''
{'app': {'name': 'MyApp', 'version': 1.0, 'environment': 'production', 'features': [{'name': 'feature1', 'enabled': True}, {'name': 'feature2', 'enabled': False}]}, 'database': {'primary': {'host': 'localhost', 'port': 5432, 'credentials': {'user': 'admin', 'password': 'secret'}}, 'secondary': {'host': 'localhost', 'port': 3306, 'credentials': {'user': 'backup', 'password': 'backup_secret'}}}}
'''

### Converting OmegaConf object back to dictionary

config_dict = OmegaConf.to_container(config)
# The above is now a regular python dictionary: Good for manipulation but not for reading and accebility
config_dict.keys() # dict_keys(['app', 'database'])
print(config_dict.app.name) # throws an error

### Converting Dictionary to OmegConf Object

config_structed = OmegaConf.structured(config_dict)
print(config_structed.app.name) # works again


############# @dataclass decorator ##################

from dataclasses import dataclass

@dataclass(order=True)
class Point:
    x: float 
    y: float = 1.0

p1 = Point(4.0, 5.0)
p2 = Point(3.0, 4.0)

print(p1)        # Output: Point(x=1.0, y=2.0)
print(p1 == p2)  # Output: False
print(p1 >= p2)  # Output: True, with order=False Runtime error

@dataclass(order=False)
class Point:
    x: float 
    y: float = 1.0
    
    def __le__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return [self.x <= other.x, self.y <= other.y]

p1 = Point(4.0, 5.0)
p2 = Point(3.0, 4.0)

print(p1)        # Output: Point(x=1.0, y=2.0)
print(p1 == p2)  # Output: False
print(p1 >= p2)  # Output: [True,True], No error despite order=False --> not of __le__ used!
print(p1 <= p2)  # Output: [False,False], __le__ definition used

# The instances of Point class can be converted to OmegaConf data type
conf = OmegaConf.structured(p1)

