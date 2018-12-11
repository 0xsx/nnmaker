# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))


project = "nnetmaker"
copyright = "2018, 0xsx"
author = "0xsx"
version = ""
release = "0.0.1"


master_doc = "index"
language = None
exclude_patterns = []
pygments_style = "sphinx"
templates_path = ["_templates"]
html_static_path = ["_static"]
source_suffix = ".rst"

# needs_sphinx = "1.0"
extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.githubpages",
]



html_theme = "alabaster"
htmlhelp_basename = "nnetmakerdoc"
html_show_sourcelink = False

# https://alabaster.readthedocs.io/en/latest/customization.html#theme-options
html_theme_options = {
  "show_relbars": False,
  "fixed_sidebar": True,
  "github_user": "0xsx",
  # "github_repo": "alabaster",
  "description": "A Python package for assisting neural network production with TensorFlow.",
}

html_sidebars = {}
