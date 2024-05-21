"""
Created by ION
"""


def get_destination_city(paths: list[list[str]]) -> str:
    to_cities = set()
    from_cities = set()
    for path in paths:
        from_cities.add(path[0])
        to_cities.add(path[1])

    destination_cities = to_cities - from_cities

    return destination_cities.pop()


paths = [
    ["London", "New York"],
    ["New York", "Lima"],
    ["Lima", "Sao Paulo"]
]


city = get_destination_city(paths)
print(f"City: {city}")
