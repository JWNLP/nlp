{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "kobert.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "mount_file_id": "123FNJj1jTRGDTrYoI-3_tpcPQs2eFDX6",
      "authorship_tag": "ABX9TyPOyycv8kOtxhHGcZwjfs0a",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU",
    "widgets": {
      "application/vnd.jupyter.widget-state+json": {
        "68c6257c680e4cba826fe7f0de9398c8": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "state": {
            "_view_name": "HBoxView",
            "_dom_classes": [],
            "_model_name": "HBoxModel",
            "_view_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_view_count": null,
            "_view_module_version": "1.5.0",
            "box_style": "",
            "layout": "IPY_MODEL_6d33c977f7d24fc68efc2fcb8f7d53df",
            "_model_module": "@jupyter-widgets/controls",
            "children": [
              "IPY_MODEL_11b6be630a9746b4b22dff778857d9dc",
              "IPY_MODEL_096d3552eef6461192c0fb80b2dd1cb6"
            ]
          }
        },
        "6d33c977f7d24fc68efc2fcb8f7d53df": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "state": {
            "_view_name": "LayoutView",
            "grid_template_rows": null,
            "right": null,
            "justify_content": null,
            "_view_module": "@jupyter-widgets/base",
            "overflow": null,
            "_model_module_version": "1.2.0",
            "_view_count": null,
            "flex_flow": null,
            "width": null,
            "min_width": null,
            "border": null,
            "align_items": null,
            "bottom": null,
            "_model_module": "@jupyter-widgets/base",
            "top": null,
            "grid_column": null,
            "overflow_y": null,
            "overflow_x": null,
            "grid_auto_flow": null,
            "grid_area": null,
            "grid_template_columns": null,
            "flex": null,
            "_model_name": "LayoutModel",
            "justify_items": null,
            "grid_row": null,
            "max_height": null,
            "align_content": null,
            "visibility": null,
            "align_self": null,
            "height": null,
            "min_height": null,
            "padding": null,
            "grid_auto_rows": null,
            "grid_gap": null,
            "max_width": null,
            "order": null,
            "_view_module_version": "1.2.0",
            "grid_template_areas": null,
            "object_position": null,
            "object_fit": null,
            "grid_auto_columns": null,
            "margin": null,
            "display": null,
            "left": null
          }
        },
        "11b6be630a9746b4b22dff778857d9dc": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "state": {
            "_view_name": "ProgressView",
            "style": "IPY_MODEL_a80a78190c9d43058e87e08bed4f581d",
            "_dom_classes": [],
            "description": "  0%",
            "_model_name": "FloatProgressModel",
            "bar_style": "danger",
            "max": 1,
            "_view_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "value": 0,
            "_view_count": null,
            "_view_module_version": "1.5.0",
            "orientation": "horizontal",
            "min": 0,
            "description_tooltip": null,
            "_model_module": "@jupyter-widgets/controls",
            "layout": "IPY_MODEL_9a41091590ea4802b4e2feeb034e35f1"
          }
        },
        "096d3552eef6461192c0fb80b2dd1cb6": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "state": {
            "_view_name": "HTMLView",
            "style": "IPY_MODEL_d5c76cfcdefe4ffb92a7853ada084a7b",
            "_dom_classes": [],
            "description": "",
            "_model_name": "HTMLModel",
            "placeholder": "​",
            "_view_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "value": " 0/1 [00:00&lt;?, ?it/s]",
            "_view_count": null,
            "_view_module_version": "1.5.0",
            "description_tooltip": null,
            "_model_module": "@jupyter-widgets/controls",
            "layout": "IPY_MODEL_4de289534ff4428f8fad52f4c2e2b311"
          }
        },
        "a80a78190c9d43058e87e08bed4f581d": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "state": {
            "_view_name": "StyleView",
            "_model_name": "ProgressStyleModel",
            "description_width": "initial",
            "_view_module": "@jupyter-widgets/base",
            "_model_module_version": "1.5.0",
            "_view_count": null,
            "_view_module_version": "1.2.0",
            "bar_color": null,
            "_model_module": "@jupyter-widgets/controls"
          }
        },
        "9a41091590ea4802b4e2feeb034e35f1": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "state": {
            "_view_name": "LayoutView",
            "grid_template_rows": null,
            "right": null,
            "justify_content": null,
            "_view_module": "@jupyter-widgets/base",
            "overflow": null,
            "_model_module_version": "1.2.0",
            "_view_count": null,
            "flex_flow": null,
            "width": null,
            "min_width": null,
            "border": null,
            "align_items": null,
            "bottom": null,
            "_model_module": "@jupyter-widgets/base",
            "top": null,
            "grid_column": null,
            "overflow_y": null,
            "overflow_x": null,
            "grid_auto_flow": null,
            "grid_area": null,
            "grid_template_columns": null,
            "flex": null,
            "_model_name": "LayoutModel",
            "justify_items": null,
            "grid_row": null,
            "max_height": null,
            "align_content": null,
            "visibility": null,
            "align_self": null,
            "height": null,
            "min_height": null,
            "padding": null,
            "grid_auto_rows": null,
            "grid_gap": null,
            "max_width": null,
            "order": null,
            "_view_module_version": "1.2.0",
            "grid_template_areas": null,
            "object_position": null,
            "object_fit": null,
            "grid_auto_columns": null,
            "margin": null,
            "display": null,
            "left": null
          }
        },
        "d5c76cfcdefe4ffb92a7853ada084a7b": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_view_name": "StyleView",
            "_model_name": "DescriptionStyleModel",
            "description_width": "",
            "_view_module": "@jupyter-widgets/base",
            "_model_module_version": "1.5.0",
            "_view_count": null,
            "_view_module_version": "1.2.0",
            "_model_module": "@jupyter-widgets/controls"
          }
        },
        "4de289534ff4428f8fad52f4c2e2b311": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "state": {
            "_view_name": "LayoutView",
            "grid_template_rows": null,
            "right": null,
            "justify_content": null,
            "_view_module": "@jupyter-widgets/base",
            "overflow": null,
            "_model_module_version": "1.2.0",
            "_view_count": null,
            "flex_flow": null,
            "width": null,
            "min_width": null,
            "border": null,
            "align_items": null,
            "bottom": null,
            "_model_module": "@jupyter-widgets/base",
            "top": null,
            "grid_column": null,
            "overflow_y": null,
            "overflow_x": null,
            "grid_auto_flow": null,
            "grid_area": null,
            "grid_template_columns": null,
            "flex": null,
            "_model_name": "LayoutModel",
            "justify_items": null,
            "grid_row": null,
            "max_height": null,
            "align_content": null,
            "visibility": null,
            "align_self": null,
            "height": null,
            "min_height": null,
            "padding": null,
            "grid_auto_rows": null,
            "grid_gap": null,
            "max_width": null,
            "order": null,
            "_view_module_version": "1.2.0",
            "grid_template_areas": null,
            "object_position": null,
            "object_fit": null,
            "grid_auto_columns": null,
            "margin": null,
            "display": null,
            "left": null
          }
        }
      }
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JWNLP/nlp/blob/main/kobert.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XAhU83fBKbhG",
        "outputId": "83620207-625f-4d6e-c3ad-f9c2cb3eefe6"
      },
      "source": [
        "!pip install mxnet\n",
        "!pip install gluonnlp pandas tqdm\n",
        "!pip install sentencepiece\n",
        "!pip install transformers==3 # 최신 버전으로 설치하면 \"Input: must be Tensor, not str\" 라는 에러 발생\n",
        "!pip install torch\n",
        "\n",
        "!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting mxnet\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/30/07/66174e78c12a3048db9039aaa09553e35035ef3a008ba3e0ed8d2aa3c47b/mxnet-1.8.0.post0-py2.py3-none-manylinux2014_x86_64.whl (46.9MB)\n",
            "\u001b[K     |████████████████████████████████| 46.9MB 66kB/s \n",
            "\u001b[?25hCollecting graphviz<0.9.0,>=0.8.1\n",
            "  Downloading https://files.pythonhosted.org/packages/53/39/4ab213673844e0c004bed8a0781a0721a3f6bb23eb8854ee75c236428892/graphviz-0.8.4-py2.py3-none-any.whl\n",
            "Requirement already satisfied: numpy<2.0.0,>1.16.0 in /usr/local/lib/python3.7/dist-packages (from mxnet) (1.19.5)\n",
            "Requirement already satisfied: requests<3,>=2.20.0 in /usr/local/lib/python3.7/dist-packages (from mxnet) (2.23.0)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.20.0->mxnet) (1.24.3)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.20.0->mxnet) (3.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.20.0->mxnet) (2020.12.5)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.20.0->mxnet) (2.10)\n",
            "Installing collected packages: graphviz, mxnet\n",
            "  Found existing installation: graphviz 0.10.1\n",
            "    Uninstalling graphviz-0.10.1:\n",
            "      Successfully uninstalled graphviz-0.10.1\n",
            "Successfully installed graphviz-0.8.4 mxnet-1.8.0.post0\n",
            "Collecting gluonnlp\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/9c/81/a238e47ccba0d7a61dcef4e0b4a7fd4473cb86bed3d84dd4fe28d45a0905/gluonnlp-0.10.0.tar.gz (344kB)\n",
            "\u001b[K     |████████████████████████████████| 348kB 5.1MB/s \n",
            "\u001b[?25hRequirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (1.1.5)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (4.41.1)\n",
            "Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from gluonnlp) (1.19.5)\n",
            "Requirement already satisfied: cython in /usr/local/lib/python3.7/dist-packages (from gluonnlp) (0.29.22)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from gluonnlp) (20.9)\n",
            "Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas) (2018.9)\n",
            "Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas) (2.8.1)\n",
            "Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->gluonnlp) (2.4.7)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas) (1.15.0)\n",
            "Building wheels for collected packages: gluonnlp\n",
            "  Building wheel for gluonnlp (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for gluonnlp: filename=gluonnlp-0.10.0-cp37-cp37m-linux_x86_64.whl size=595684 sha256=bb91ed7c018c27ff18f731d27b021bc0079b02453dc2d61d9eec520605810512\n",
            "  Stored in directory: /root/.cache/pip/wheels/37/65/52/63032864a0f31a08b9a88569f803b5bafac8abd207fd7f7534\n",
            "Successfully built gluonnlp\n",
            "Installing collected packages: gluonnlp\n",
            "Successfully installed gluonnlp-0.10.0\n",
            "Collecting sentencepiece\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/f5/99/e0808cb947ba10f575839c43e8fafc9cc44e4a7a2c8f79c60db48220a577/sentencepiece-0.1.95-cp37-cp37m-manylinux2014_x86_64.whl (1.2MB)\n",
            "\u001b[K     |████████████████████████████████| 1.2MB 5.1MB/s \n",
            "\u001b[?25hInstalling collected packages: sentencepiece\n",
            "Successfully installed sentencepiece-0.1.95\n",
            "Collecting transformers==3\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/9c/35/1c3f6e62d81f5f0daff1384e6d5e6c5758682a8357ebc765ece2b9def62b/transformers-3.0.0-py3-none-any.whl (754kB)\n",
            "\u001b[K     |████████████████████████████████| 757kB 6.5MB/s \n",
            "\u001b[?25hRequirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from transformers==3) (20.9)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from transformers==3) (1.19.5)\n",
            "Requirement already satisfied: sentencepiece in /usr/local/lib/python3.7/dist-packages (from transformers==3) (0.1.95)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers==3) (2019.12.20)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.7/dist-packages (from transformers==3) (4.41.1)\n",
            "Collecting sacremoses\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/08/cd/342e584ee544d044fb573ae697404ce22ede086c9e87ce5960772084cad0/sacremoses-0.0.44.tar.gz (862kB)\n",
            "\u001b[K     |████████████████████████████████| 870kB 10.7MB/s \n",
            "\u001b[?25hRequirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers==3) (3.0.12)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers==3) (2.23.0)\n",
            "Collecting tokenizers==0.8.0-rc4\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/f7/82/0e82a95bd9db2b32569500cc1bb47aa7c4e0f57aa5e35cceba414096917b/tokenizers-0.8.0rc4-cp37-cp37m-manylinux1_x86_64.whl (3.0MB)\n",
            "\u001b[K     |████████████████████████████████| 3.0MB 12.0MB/s \n",
            "\u001b[?25hRequirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->transformers==3) (2.4.7)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers==3) (1.15.0)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers==3) (7.1.2)\n",
            "Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers==3) (1.0.1)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers==3) (1.24.3)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers==3) (2.10)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers==3) (2020.12.5)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers==3) (3.0.4)\n",
            "Building wheels for collected packages: sacremoses\n",
            "  Building wheel for sacremoses (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for sacremoses: filename=sacremoses-0.0.44-cp37-none-any.whl size=886084 sha256=668377baf07c0e4bec014071860d8cabbb213c2d70172b743a1e2c15587686df\n",
            "  Stored in directory: /root/.cache/pip/wheels/3e/fb/c0/13ab4d63d537658f448366744654323077c4d90069b6512f3c\n",
            "Successfully built sacremoses\n",
            "Installing collected packages: sacremoses, tokenizers, transformers\n",
            "Successfully installed sacremoses-0.0.44 tokenizers-0.8.0rc4 transformers-3.0.0\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.7/dist-packages (1.8.1+cu101)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch) (3.7.4.3)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from torch) (1.19.5)\n",
            "Collecting git+https://****@github.com/SKTBrain/KoBERT.git@master\n",
            "  Cloning https://****@github.com/SKTBrain/KoBERT.git (to revision master) to /tmp/pip-req-build-wlr3bgt6\n",
            "  Running command git clone -q 'https://****@github.com/SKTBrain/KoBERT.git' /tmp/pip-req-build-wlr3bgt6\n",
            "Building wheels for collected packages: kobert\n",
            "  Building wheel for kobert (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for kobert: filename=kobert-0.1.2-cp37-none-any.whl size=12708 sha256=049e2374959b124069617289daa360f694080fafc8e5e8cde65c4c54e8201295\n",
            "  Stored in directory: /tmp/pip-ephem-wheel-cache-9wkwlgeg/wheels/a2/b0/41/435ee4e918f91918be41529283c5ff86cd010f02e7525aecf3\n",
            "Successfully built kobert\n",
            "Installing collected packages: kobert\n",
            "Successfully installed kobert-0.1.2\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mq9fDZpdj3FE"
      },
      "source": [
        "import torch\n",
        "from torch import nn\n",
        "import torch.nn.functional as F\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "import gluonnlp as nlp\n",
        "import numpy as np\n",
        "from tqdm import tqdm, tqdm_notebook\n",
        "\n",
        "from kobert.utils import get_tokenizer\n",
        "from kobert.pytorch_kobert import get_pytorch_kobert_model\n",
        "\n",
        "from transformers import AdamW\n",
        "from transformers.optimization import get_cosine_schedule_with_warmup"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "NTWPHqIIj0Fa"
      },
      "source": [
        "##GPU 사용 시\n",
        "device = torch.device(\"cuda:0\")\n"
      ],
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "W0So-8_XkLVq",
        "outputId": "799a696b-33d2-42fd-e06c-701204319931"
      },
      "source": [
        "bertmodel, vocab = get_pytorch_kobert_model()"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[██████████████████████████████████████████████████]\n",
            "[██████████████████████████████████████████████████]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 415
        },
        "id": "ul5mJOd4K4J6",
        "outputId": "5f533509-b818-4d0c-c0f2-2f9d0a3765cd"
      },
      "source": [
        "# 학습용 데이터셋 불러오기\n",
        "import pandas as pd\n",
        "\n",
        "td_hate = pd.read_csv('/content/drive/MyDrive/케글/train.hate.csv')\n",
        "td_hate.head\n",
        "td_hate\n",
        "\n"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>comments</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>(현재 호텔주인 심정) 아18 난 마른하늘에 날벼락맞고 호텔망하게생겼는데 누군 계속...</td>\n",
              "      <td>hate</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>....한국적인 미인의 대표적인 분...너무나 곱고아름다운모습...그모습뒤의 슬픔을...</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>...못된 넘들...남의 고통을 즐겼던 넘들..이젠 마땅한 처벌을 받아야지..,그래...</td>\n",
              "      <td>hate</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1,2화 어설펐는데 3,4화 지나서부터는 갈수록 너무 재밌던데</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1. 사람 얼굴 손톱으로 긁은것은 인격살해이고2. 동영상이 몰카냐? 메걸리안들 생각...</td>\n",
              "      <td>hate</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7891</th>\n",
              "      <td>힘내세요~ 응원합니다!!</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7892</th>\n",
              "      <td>힘내세요~~삼가 고인의 명복을 빕니다..</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7893</th>\n",
              "      <td>힘내세용 ^^ 항상 응원합니닷 ^^ !</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7894</th>\n",
              "      <td>힘내소...연기로 답해요.나도 53살 인데 이런일 저런일 다 있더라구요.인격을 믿습...</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7895</th>\n",
              "      <td>힘들면 관뒀어야지 그게 현명한거다</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>7896 rows × 2 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "                                               comments label\n",
              "0     (현재 호텔주인 심정) 아18 난 마른하늘에 날벼락맞고 호텔망하게생겼는데 누군 계속...  hate\n",
              "1     ....한국적인 미인의 대표적인 분...너무나 곱고아름다운모습...그모습뒤의 슬픔을...  none\n",
              "2     ...못된 넘들...남의 고통을 즐겼던 넘들..이젠 마땅한 처벌을 받아야지..,그래...  hate\n",
              "3                    1,2화 어설펐는데 3,4화 지나서부터는 갈수록 너무 재밌던데  none\n",
              "4     1. 사람 얼굴 손톱으로 긁은것은 인격살해이고2. 동영상이 몰카냐? 메걸리안들 생각...  hate\n",
              "...                                                 ...   ...\n",
              "7891                                      힘내세요~ 응원합니다!!  none\n",
              "7892                             힘내세요~~삼가 고인의 명복을 빕니다..  none\n",
              "7893                              힘내세용 ^^ 항상 응원합니닷 ^^ !  none\n",
              "7894  힘내소...연기로 답해요.나도 53살 인데 이런일 저런일 다 있더라구요.인격을 믿습...  none\n",
              "7895                                 힘들면 관뒀어야지 그게 현명한거다  none\n",
              "\n",
              "[7896 rows x 2 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rnigOrDaLgt6",
        "outputId": "ce2050a1-3ab2-49ea-f431-ae07ff687e8b"
      },
      "source": [
        "dev_hate = pd.read_csv('/content/drive/MyDrive/케글/dev.hate.csv')\n",
        "dev_hate.head\n",
        "dev_hate.info()"
      ],
      "execution_count": 69,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 471 entries, 0 to 470\n",
            "Data columns (total 2 columns):\n",
            " #   Column    Non-Null Count  Dtype \n",
            "---  ------    --------------  ----- \n",
            " 0   comments  471 non-null    object\n",
            " 1   label     471 non-null    object\n",
            "dtypes: object(2)\n",
            "memory usage: 7.5+ KB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 319
        },
        "id": "sb-VbJNKL--T",
        "outputId": "ebd084fc-1d31-4618-9b68-391c02e45bb8"
      },
      "source": [
        "td_hate['label'].value_counts().plot(kind='bar')"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f2273b51f10>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 11
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "802kJIhsMPGn",
        "outputId": "f2b16b1d-0aab-4d0a-c471-fbf656170561"
      },
      "source": [
        "print(td_hate.groupby('label').size().reset_index(name='count'))"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "       label  count\n",
            "0       hate   1911\n",
            "1       none   3486\n",
            "2  offensive   2499\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Jbh2cpLVMTTu",
        "outputId": "3a8e2f94-78f9-42ee-98f0-cd01a7387dc0"
      },
      "source": [
        "td_hate['comments']= td_hate['comments'].str.replace(\"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]\",\"\")\n",
        "td_hate['comments'] = td_hate['comments'].str.replace('^ +', \"\")\n",
        "td_hate['comments'].replace('', np.nan, inplace=True)\n",
        "print(td_hate.isnull().sum())\n"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "comments    0\n",
            "label       0\n",
            "dtype: int64\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sj_Hu3TOz5Ic",
        "outputId": "b171e41d-faa9-44e4-83d4-4c635ecf808e"
      },
      "source": [
        "dev_hate['comments']= dev_hate['comments'].str.replace(\"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]\",\"\")\n",
        "dev_hate['comments'] = dev_hate['comments'].str.replace('^ +', \"\")\n",
        "dev_hate['comments'].replace('', np.nan, inplace=True)\n",
        "print(dev_hate.isnull().sum())"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "comments    0\n",
            "label       0\n",
            "dtype: int64\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BUYPXUETMjMJ",
        "outputId": "b74ec9c3-cfc2-4b30-e082-4fa4cafbe5bc"
      },
      "source": [
        "dev_hate.info()\n",
        "print('')\n",
        "print('----')\n",
        "td_hate.info()"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 471 entries, 0 to 470\n",
            "Data columns (total 2 columns):\n",
            " #   Column    Non-Null Count  Dtype \n",
            "---  ------    --------------  ----- \n",
            " 0   comments  471 non-null    object\n",
            " 1   label     471 non-null    object\n",
            "dtypes: object(2)\n",
            "memory usage: 7.5+ KB\n",
            "\n",
            "----\n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 7896 entries, 0 to 7895\n",
            "Data columns (total 2 columns):\n",
            " #   Column    Non-Null Count  Dtype \n",
            "---  ------    --------------  ----- \n",
            " 0   comments  7896 non-null   object\n",
            " 1   label     7896 non-null   object\n",
            "dtypes: object(2)\n",
            "memory usage: 123.5+ KB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 202
        },
        "id": "1Ev5AltXSxPJ",
        "outputId": "fe1bdf9e-ea03-47eb-efd2-f7118aecb6ae"
      },
      "source": [
        "# 욕 강도 추출\n",
        "data1 = td_hate.loc[td_hate['label'] == 'none']\n",
        "data2 = td_hate.loc[td_hate['label'] == 'offensive']\n",
        "data3 = td_hate.loc[td_hate['label'] == 'hate']\n",
        "\n",
        "new_td_hate = data1.append([data2,data3],sort=False)\n",
        "new_td_hate = pd.DataFrame(new_td_hate)\n",
        "new_td_hate.head()\n",
        "\n",
        "new_dev_hate = data1.append([data2,data3],sort=False)\n",
        "new_dev_hate = pd.DataFrame(new_dev_hate)\n",
        "new_dev_hate.head()\n",
        "\n"
      ],
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>comments</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>한국적인 미인의 대표적인 분너무나 곱고아름다운모습그모습뒤의 슬픔을 미처 알지못했네요ㅠ</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>화 어설펐는데 화 지나서부터는 갈수록 너무 재밌던데</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>진짜 이승기랑 비교된다</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>년뒤 윤서인은 분명히 재평가될것임 말하나하나가 틀린게없음</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>살 차이가 넘을텐데 부부라고 무슨 내용인지 긍금하네</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                                           comments label\n",
              "1   한국적인 미인의 대표적인 분너무나 곱고아름다운모습그모습뒤의 슬픔을 미처 알지못했네요ㅠ  none\n",
              "3                      화 어설펐는데 화 지나서부터는 갈수록 너무 재밌던데  none\n",
              "5                                      진짜 이승기랑 비교된다  none\n",
              "7                   년뒤 윤서인은 분명히 재평가될것임 말하나하나가 틀린게없음  none\n",
              "10                     살 차이가 넘을텐데 부부라고 무슨 내용인지 긍금하네  none"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "upeohJtfWgkT",
        "outputId": "6127ab94-7060-4dfb-9cd7-a16b31326d94"
      },
      "source": [
        "# td_hate 욕 강도 라벨링\n",
        "\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "\n",
        "encoder = LabelEncoder()\n",
        "encoder.fit(new_td_hate['label'])\n",
        "new_td_hate['label'] = encoder.fit_transform(new_td_hate['label'])\n",
        "new_td_hate.info()\n",
        "\n"
      ],
      "execution_count": 72,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 7896 entries, 1 to 7887\n",
            "Data columns (total 2 columns):\n",
            " #   Column    Non-Null Count  Dtype \n",
            "---  ------    --------------  ----- \n",
            " 0   comments  7896 non-null   object\n",
            " 1   label     7896 non-null   int64 \n",
            "dtypes: int64(1), object(1)\n",
            "memory usage: 185.1+ KB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 202
        },
        "id": "HfKap21A7Gns",
        "outputId": "0ceec2a9-7de3-4f31-ed4f-afcb61ae71fd"
      },
      "source": [
        "# dev_hate 욕 강도 추출\n",
        "data1 = dev_hate.loc[td_hate['label'] == 'none']\n",
        "data2 = dev_hate.loc[td_hate['label'] == 'offensive']\n",
        "data3 = dev_hate.loc[td_hate['label'] == 'hate']\n",
        "\n",
        "new_dev_hate = data1.append([data2,data3],sort=False)\n",
        "new_dev_hate = pd.DataFrame(new_dev_hate)\n",
        "new_dev_hate.head()"
      ],
      "execution_count": 68,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>comments</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>지현우 나쁜놈</td>\n",
              "      <td>offensive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>설마 ㅈ 현정 작가 아니지</td>\n",
              "      <td>hate</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>일베충들 ㅂㄷ거리는것봐라 ㅉㅉ</td>\n",
              "      <td>hate</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>강부자 선생님 전미선 비보에 오열을 하셨다니 눈물이 나네요 힘내세요</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>이경규가 이런거보면 세련되긴함 저 나이에 차은우 누가 알꼬 아무리 잘생겼다해도 배워...</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                                             comments      label\n",
              "1                                             지현우 나쁜놈  offensive\n",
              "3                                      설마 ㅈ 현정 작가 아니지       hate\n",
              "5                                    일베충들 ㅂㄷ거리는것봐라 ㅉㅉ       hate\n",
              "7               강부자 선생님 전미선 비보에 오열을 하셨다니 눈물이 나네요 힘내세요       none\n",
              "10  이경규가 이런거보면 세련되긴함 저 나이에 차은우 누가 알꼬 아무리 잘생겼다해도 배워...       none"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 68
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LUnnvl8K7UG7",
        "outputId": "915b4f36-1cc2-4ba0-d084-a0adaeff33c0"
      },
      "source": [
        "encoder = LabelEncoder()\n",
        "encoder.fit(new_dev_hate['label'])\n",
        "new_dev_hate['label'] = encoder.transform(new_dev_hate['label'])\n",
        "new_dev_hate.head()\n",
        "new_dev_hate.info()"
      ],
      "execution_count": 66,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Int64Index: 471 entries, 1 to 470\n",
            "Data columns (total 2 columns):\n",
            " #   Column    Non-Null Count  Dtype \n",
            "---  ------    --------------  ----- \n",
            " 0   comments  471 non-null    object\n",
            " 1   label     471 non-null    int64 \n",
            "dtypes: int64(1), object(1)\n",
            "memory usage: 11.0+ KB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ii-QiWh8Xdxf",
        "outputId": "3dd6340d-e7e1-418e-a6e3-fb8e56e631c1"
      },
      "source": [
        "# 라벨링 된 욕 강도 매핑  --> {0: 'hate', 1: 'none', 2: 'offensive'}\n",
        "\n",
        "mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))\n",
        "mapping"
      ],
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{0: 'hate', 1: 'none', 2: 'offensive'}"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FdgApfbW7SWD",
        "outputId": "4bd46f0f-4b21-40b2-8cc7-bbea78ff6dd9"
      },
      "source": [
        "# 기본 Bert tokenizer 사용\n",
        "tokenizer = get_tokenizer()\n",
        "tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)"
      ],
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "using cached model\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "d-BOqyvqMrPm"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HDdH-onUMNR9"
      },
      "source": [
        " #에러 \n",
        " '''\n",
        " class BERTDataset(Dataset): 에러      \n",
        "        # self.labels = [np.int32(i[label_idx]) for i in dataset] 로 하니까 \n",
        "        \n",
        "        data_train = BERTDataset(td_hate, 0, 1, tok, max_len, True, False) 에러1발생\n",
        "        # 에러1:  ValueError: invalid literal for int() with base 10: 'o'\n",
        "\n",
        "        # 그래서 np.int32(i[label_idx]) --> np.str(i[label_idx]) 스트링 형태로 형변환\n",
        "        # self.labels = [np.str(i[label_idx]) for i in dataset]\n",
        "'''"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9R4vXPZjmUqC"
      },
      "source": [
        "class BERTDataset(Dataset):\n",
        "    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,\n",
        "                 pad, pair):\n",
        "        transform = nlp.data.BERTSentenceTransform(\n",
        "            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)\n",
        "\n",
        "        self.sentences = [transform([i[sent_idx]]) for i in dataset]\n",
        "        \n",
        "        # 에러 ValueError: invalid literal for int() with base 10: 'o'\n",
        "        # np.int32(i[label_idx]) --> np.str(i[label_idx])\n",
        "        self.labels = [np.str(i[label_idx]) for i in dataset]\n",
        "\n",
        "    def __getitem__(self, i):\n",
        "        return (self.sentences[i] + (self.labels[i], ))\n",
        "\n",
        "    def __len__(self):\n",
        "        return (len(self.labels))\n",
        "\n"
      ],
      "execution_count": 73,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lwGK8Awjnukn"
      },
      "source": [
        "# Setting parameters\n",
        "max_len = 64 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음\n",
        "batch_size = 64\n",
        "warmup_ratio = 0.1\n",
        "num_epochs = 5\n",
        "max_grad_norm = 1\n",
        "log_interval = 200\n",
        "learning_rate = 5e-5"
      ],
      "execution_count": 74,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UHgWRZf-oNgE",
        "outputId": "9de45db3-f283-40cb-a358-6260ef9a8a44"
      },
      "source": [
        "# self.labels = [np.int32(i[label_idx]) for i in dataset\n",
        "#에러 :ValueError: invalid literal for int() with base 10: 'o'\n",
        "data_train = BERTDataset(td_hate, 0, 1, tok, max_len, True, False)\n",
        "data_test = BERTDataset(dev_hate, 0, 1, tok, max_len, True, False)\n",
        "\n",
        "# pytorch용 DataLoader 사용\n",
        "train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)\n",
        "dev_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)\n"
      ],
      "execution_count": 75,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:477: UserWarning: This DataLoader will create 5 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.\n",
            "  cpuset_checked))\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lXoNElCYn9fI"
      },
      "source": [
        "class BERTClassifier(nn.Module):\n",
        "    def __init__(self,\n",
        "                 bert,\n",
        "                 hidden_size = 768,\n",
        "                 num_classes = 3, # softmax 사용 <- binary일 경우는 2\n",
        "                 dr_rate=None,\n",
        "                 params=None):\n",
        "        super(BERTClassifier, self).__init__()\n",
        "        self.bert = bert\n",
        "        self.dr_rate = dr_rate\n",
        "                 \n",
        "        self.classifier = nn.Linear(hidden_size , num_classes)\n",
        "        if dr_rate:\n",
        "            self.dropout = nn.Dropout(p=dr_rate)\n",
        "    \n",
        "    def gen_attention_mask(self, token_ids, valid_length):\n",
        "        attention_mask = torch.zeros_like(token_ids)\n",
        "        for i, v in enumerate(valid_length):\n",
        "            attention_mask[i][:v] = 1\n",
        "        return attention_mask.float()\n",
        "\n",
        "    def forward(self, token_ids, valid_length, segment_ids):\n",
        "        attention_mask = self.gen_attention_mask(token_ids, valid_length)\n",
        "        \n",
        "        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))\n",
        "        if self.dr_rate:\n",
        "            out = self.dropout(pooler)\n",
        "        return self.classifier(out)"
      ],
      "execution_count": 76,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eBtMJahq-Su2"
      },
      "source": [
        "model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)"
      ],
      "execution_count": 77,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SKI-F-B9-ZoI"
      },
      "source": [
        "# Prepare optimizer and schedule (linear warmup and decay)\n",
        "no_decay = ['bias', 'LayerNorm.weight']\n",
        "optimizer_grouped_parameters = [\n",
        "    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},\n",
        "    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}\n",
        "]\n"
      ],
      "execution_count": 78,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fX14aZo1-cN7"
      },
      "source": [
        "# 옵티마이저 선언\n",
        "optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)\n",
        "loss_fn = nn.CrossEntropyLoss() # softmax용 Loss Function 정하기 <- binary classification도 해당 loss function 사용 가능\n",
        "\n",
        "t_total = len(train_dataloader) * num_epochs\n",
        "warmup_step = int(t_total * warmup_ratio)\n",
        "\n",
        "scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)"
      ],
      "execution_count": 79,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Kok4GEbe-f5o"
      },
      "source": [
        "# 학습 평가 지표인 accuracy 계산 -> 얼마나 타겟값을 많이 맞추었는가\n",
        "def calc_accuracy(X,Y):\n",
        "    max_vals, max_indices = torch.max(X, 1)\n",
        "    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]\n",
        "    return train_acc"
      ],
      "execution_count": 80,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "drKF-a8dNssD"
      },
      "source": [
        "# 에러2 :   #AttributeError: 'tuple' object has no attribute 'long'\n",
        "#        label = label.long().to(device)\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 375,
          "referenced_widgets": [
            "68c6257c680e4cba826fe7f0de9398c8",
            "6d33c977f7d24fc68efc2fcb8f7d53df",
            "11b6be630a9746b4b22dff778857d9dc",
            "096d3552eef6461192c0fb80b2dd1cb6",
            "a80a78190c9d43058e87e08bed4f581d",
            "9a41091590ea4802b4e2feeb034e35f1",
            "d5c76cfcdefe4ffb92a7853ada084a7b",
            "4de289534ff4428f8fad52f4c2e2b311"
          ]
        },
        "id": "vV5Fr0NT-iqr",
        "outputId": "15c67c33-4e59-4b16-86a2-7743b8837a61"
      },
      "source": [
        "# 모델 학습 시작\n",
        "for e in range(num_epochs):\n",
        "    train_acc = 0.0\n",
        "    test_acc = 0.0\n",
        "    model.train()\n",
        "    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):\n",
        "        optimizer.zero_grad()\n",
        "        token_ids = token_ids.long().to(device)\n",
        "        segment_ids = segment_ids.long().to(device)\n",
        "        valid_length= valid_length\n",
        "\n",
        "        #AttributeError: 'tuple' object has no attribute 'long'\n",
        "        label = label.long().to(device)\n",
        "        \n",
        "        out = model(token_ids, valid_length, segment_ids)\n",
        "        loss = loss_fn(out, label)\n",
        "        loss.backward()\n",
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # gradient clipping\n",
        "        optimizer.step()\n",
        "        scheduler.step()  # Update learning rate schedule\n",
        "        train_acc += calc_accuracy(out, label)\n",
        "        if batch_id % log_interval == 0:\n",
        "            print(\"epoch {} batch id {} loss {} train acc {}\".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))\n",
        "    print(\"epoch {} train acc {}\".format(e+1, train_acc / (batch_id+1)))"
      ],
      "execution_count": 83,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:6: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0\n",
            "Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`\n",
            "  \n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "display_data",
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "68c6257c680e4cba826fe7f0de9398c8",
              "version_minor": 0,
              "version_major": 2
            },
            "text/plain": [
              "HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:477: UserWarning: This DataLoader will create 5 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.\n",
            "  cpuset_checked))\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "error",
          "ename": "AttributeError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-83-2f7ed119a770>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      9\u001b[0m         \u001b[0msegment_ids\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msegment_ids\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlong\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mto\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdevice\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m         \u001b[0mvalid_length\u001b[0m\u001b[0;34m=\u001b[0m \u001b[0mvalid_length\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 11\u001b[0;31m         \u001b[0mlabel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlabel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlong\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mto\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdevice\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     12\u001b[0m         \u001b[0mout\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtoken_ids\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalid_length\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msegment_ids\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     13\u001b[0m         \u001b[0mloss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mloss_fn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mout\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabel\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mAttributeError\u001b[0m: 'tuple' object has no attribute 'long'"
          ]
        }
      ]
    }
  ]
}
