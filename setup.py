#!/usr/bin/env python3

import setuptools


def parse_requirements_file(path):
  return [line.rstrip() for line in open(path, "r") if not line.startswith("#")]


main_requirements = parse_requirements_file("requirements/main.txt")
print(main_requirements)
dev_requirements = parse_requirements_file("requirements/dev.txt")
envs_requirements = parse_requirements_file("requirements/envs.txt")

if __name__ == "__main__":
  setuptools.setup(
      install_requires=main_requirements,
      extra_require={
          "dev": dev_requirements,
          "envs": envs_requirements
      },
  )
