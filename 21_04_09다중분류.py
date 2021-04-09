{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "21-04-09다중분류.ipynb",
      "provenance": [],
      "collapsed_sections": [
        "fl3Dj47ormxd",
        "Lp6e73UsssI9",
        "Qk5UI14It5Mm",
        "bM2qM_4mxCE1",
        "lI30hrw6xQXc"
      ],
      "mount_file_id": "14YUwjyPFq8UkkD6Jq2RhWi_-vjI9oeH6",
      "authorship_tag": "ABX9TyOYsL9DHY7ew+1jYYbHFMGo",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JWNLP/nlp/blob/main/21_04_09%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3EWP_4a8pYnb",
        "outputId": "e5ac0e58-f0b7-43fa-a0b4-285a3911b8eb"
      },
      "source": [
        "# Hugging Face의 트랜스포머 모델을 설치\n",
        "!pip install transformers"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: transformers in /usr/local/lib/python3.7/dist-packages (4.5.0)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (1.19.5)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers) (3.0.12)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.7/dist-packages (from transformers) (4.41.1)\n",
            "Requirement already satisfied: importlib-metadata; python_version < \"3.8\" in /usr/local/lib/python3.7/dist-packages (from transformers) (3.8.1)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from transformers) (20.9)\n",
            "Requirement already satisfied: tokenizers<0.11,>=0.10.1 in /usr/local/lib/python3.7/dist-packages (from transformers) (0.10.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (2019.12.20)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers) (2.23.0)\n",
            "Requirement already satisfied: sacremoses in /usr/local/lib/python3.7/dist-packages (from transformers) (0.0.44)\n",
            "Requirement already satisfied: typing-extensions>=3.6.4; python_version < \"3.8\" in /usr/local/lib/python3.7/dist-packages (from importlib-metadata; python_version < \"3.8\"->transformers) (3.7.4.3)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata; python_version < \"3.8\"->transformers) (3.4.1)\n",
            "Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->transformers) (2.4.7)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2.10)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (1.24.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2020.12.5)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (3.0.4)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (7.1.2)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (1.15.0)\n",
            "Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from sacremoses->transformers) (1.0.1)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-IKYE0KRqMLm"
      },
      "source": [
        "import tensorflow as tf\n",
        "import torch\n",
        "\n",
        "from transformers import BertTokenizer\n",
        "from transformers import BertForSequenceClassification, AdamW, BertConfig\n",
        "from transformers import get_linear_schedule_with_warmup\n",
        "from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler\n",
        "from keras.preprocessing.sequence import pad_sequences\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import random\n",
        "import time\n",
        "import datetime"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "58oJRt7OqNFm",
        "outputId": "341817c7-00b1-47e8-fbf8-b79ba56b8e16"
      },
      "source": [
        "train = pd.read_csv(\"/content/케글/train.hate.csv\", sep=',')\n",
        "dev = pd.read_csv(\"/content/케글/dev.hate.csv\", sep=',')\n",
        "test = pd.read_csv(\"/content/케글/test.hate.no_label.csv\", sep=',')\n",
        "\n",
        "print(train.info)\n",
        "print(dev.info)\n",
        "print(test.info)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<bound method DataFrame.info of                                                comments label\n",
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
            "[7896 rows x 2 columns]>\n",
            "<bound method DataFrame.info of                                               comments      label\n",
            "0                          송중기 시대극은 믿고본다. 첫회 신선하고 좋았다.       none\n",
            "1                                              지현우 나쁜놈  offensive\n",
            "2           알바쓰고많이만들면되지 돈욕심없으면골목식당왜나온겨 기댕기게나하고 산에가서팔어라       hate\n",
            "3                                     설마 ㅈ 현정 작가 아니지??       hate\n",
            "4    이미자씨 송혜교씨 돈이 그리 많으면 탈세말고 그돈으로 평소에 불우이웃에게 기부도 좀...  offensive\n",
            "..                                                 ...        ...\n",
            "466                                  지현우 범죄 저지르지 않았나요?  offensive\n",
            "467                                    여자인생 망칠 일 있나 ㅋㅋ       hate\n",
            "468            근데 전라도에서 사고가 났는데 굳이 서울까지 와서 병원에 가느 이유는?  offensive\n",
            "469  할매젖x, 뱃살x, 몸매 s라인, 유륜은 적당해야됨(너무크거나 너무 작아도 x), ...       hate\n",
            "470  남자가 잘못한거라면... 반성도 없다면...나였다면 ... 여자처럼 아주 못되게 할...       none\n",
            "\n",
            "[471 rows x 2 columns]>\n",
            "<bound method DataFrame.info of                                               comments\n",
            "0         ㅋㅋㅋㅋ 그래도 조아해주는 팬들 많아서 좋겠다 ㅠㅠ 니들은 온유가 안만져줌 ㅠㅠ\n",
            "1                                        둘다 넘 좋다~행복하세요\n",
            "2                 근데 만원이하는 현금결제만 하라고 써놓은집 우리나라에 엄청 많은데\n",
            "3                원곡생각하나도 안나고 러블리즈 신곡나온줄!!! 너무 예쁘게 잘봤어요\n",
            "4                                   장현승 얘도 참 이젠 짠하다...\n",
            "..                                                 ...\n",
            "969                     대박 게스트... 꼭 봐야징~ 컨셉이 바뀌니깐 재미지넹\n",
            "970  성형으로 다 뜯어고쳐놓고 예쁜척. 성형 전 니 얼굴 다 알고있다. 순자처럼 된장냄새...\n",
            "971  분위기는 비슷하다만 전혀다른 전개던데 무슨ㅋㅋㄱ 우리나라사람들은 분위기만 비슷하면 ...\n",
            "972                               입에 손가릭이 10개 있으니 징그럽다\n",
            "973                              난 조보아 이뻐서 보는데 백종원 관심무\n",
            "\n",
            "[974 rows x 1 columns]>\n"
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
          "height": 355
        },
        "id": "Vphhou47qj9Q",
        "outputId": "5e998901-d609-4da9-95df-8845199ca993"
      },
      "source": [
        "train.head(10)"
      ],
      "execution_count": null,
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
              "      <th>5</th>\n",
              "      <td>10+8 진짜 이승기랑 비교된다</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>100년안에 남녀간 성전쟁 한번 크게 치룬 후 일부다처제, 여성의 정치참여 금지, ...</td>\n",
              "      <td>hate</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>10년뒤 윤서인은 분명히 재평가될것임. 말하나하나가 틀린게없음</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>10년만에 재미를 느끼는 프로였는데왜 니들때문에 폐지를해야되냐</td>\n",
              "      <td>offensive</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>10년차방탄팬인데 우리방탄처럼 성공은못하겠지만 일단 방탄의 부하가되고싶다는거니 이름...</td>\n",
              "      <td>offensive</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                                            comments      label\n",
              "0  (현재 호텔주인 심정) 아18 난 마른하늘에 날벼락맞고 호텔망하게생겼는데 누군 계속...       hate\n",
              "1  ....한국적인 미인의 대표적인 분...너무나 곱고아름다운모습...그모습뒤의 슬픔을...       none\n",
              "2  ...못된 넘들...남의 고통을 즐겼던 넘들..이젠 마땅한 처벌을 받아야지..,그래...       hate\n",
              "3                 1,2화 어설펐는데 3,4화 지나서부터는 갈수록 너무 재밌던데       none\n",
              "4  1. 사람 얼굴 손톱으로 긁은것은 인격살해이고2. 동영상이 몰카냐? 메걸리안들 생각...       hate\n",
              "5                                  10+8 진짜 이승기랑 비교된다       none\n",
              "6  100년안에 남녀간 성전쟁 한번 크게 치룬 후 일부다처제, 여성의 정치참여 금지, ...       hate\n",
              "7                 10년뒤 윤서인은 분명히 재평가될것임. 말하나하나가 틀린게없음       none\n",
              "8                 10년만에 재미를 느끼는 프로였는데왜 니들때문에 폐지를해야되냐  offensive\n",
              "9  10년차방탄팬인데 우리방탄처럼 성공은못하겠지만 일단 방탄의 부하가되고싶다는거니 이름...  offensive"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 74
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8mAxqTo-q3lr",
        "outputId": "020ffadd-261a-4bad-f7cd-354ba13baf19"
      },
      "source": [
        "# 리뷰 문장 추출\n",
        "sentences = train['comments']\n",
        "sentences[:10]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0    (현재 호텔주인 심정) 아18 난 마른하늘에 날벼락맞고 호텔망하게생겼는데 누군 계속...\n",
              "1    ....한국적인 미인의 대표적인 분...너무나 곱고아름다운모습...그모습뒤의 슬픔을...\n",
              "2    ...못된 넘들...남의 고통을 즐겼던 넘들..이젠 마땅한 처벌을 받아야지..,그래...\n",
              "3                   1,2화 어설펐는데 3,4화 지나서부터는 갈수록 너무 재밌던데\n",
              "4    1. 사람 얼굴 손톱으로 긁은것은 인격살해이고2. 동영상이 몰카냐? 메걸리안들 생각...\n",
              "5                                    10+8 진짜 이승기랑 비교된다\n",
              "6    100년안에 남녀간 성전쟁 한번 크게 치룬 후 일부다처제, 여성의 정치참여 금지, ...\n",
              "7                   10년뒤 윤서인은 분명히 재평가될것임. 말하나하나가 틀린게없음\n",
              "8                   10년만에 재미를 느끼는 프로였는데왜 니들때문에 폐지를해야되냐\n",
              "9    10년차방탄팬인데 우리방탄처럼 성공은못하겠지만 일단 방탄의 부하가되고싶다는거니 이름...\n",
              "Name: comments, dtype: object"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 75
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bqm0FZ86q74i",
        "outputId": "7c4dc07e-18d2-488e-d783-3b12994d862e"
      },
      "source": [
        "# BERT의 입력 형식에 맞게 변환\n",
        "sentences = [\"[CLS] \" + str(sentence) + \" [SEP]\" for sentence in sentences]\n",
        "sentences[:10]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['[CLS] (현재 호텔주인 심정) 아18 난 마른하늘에 날벼락맞고 호텔망하게생겼는데 누군 계속 추모받네.... [SEP]',\n",
              " '[CLS] ....한국적인 미인의 대표적인 분...너무나 곱고아름다운모습...그모습뒤의 슬픔을 미처 알지못했네요ㅠ [SEP]',\n",
              " '[CLS] ...못된 넘들...남의 고통을 즐겼던 넘들..이젠 마땅한 처벌을 받아야지..,그래야, 공정한 사회지...심은대로 거두거라... [SEP]',\n",
              " '[CLS] 1,2화 어설펐는데 3,4화 지나서부터는 갈수록 너무 재밌던데 [SEP]',\n",
              " '[CLS] 1. 사람 얼굴 손톱으로 긁은것은 인격살해이고2. 동영상이 몰카냐? 메걸리안들 생각이 없노 [SEP]',\n",
              " '[CLS] 10+8 진짜 이승기랑 비교된다 [SEP]',\n",
              " '[CLS] 100년안에 남녀간 성전쟁 한번 크게 치룬 후 일부다처제, 여성의 정치참여 금지, 여성 투표권 삭제가 세계의 공통문화로 자리잡을듯. 암탉이 너무 울어댐. [SEP]',\n",
              " '[CLS] 10년뒤 윤서인은 분명히 재평가될것임. 말하나하나가 틀린게없음 [SEP]',\n",
              " '[CLS] 10년만에 재미를 느끼는 프로였는데왜 니들때문에 폐지를해야되냐 [SEP]',\n",
              " '[CLS] 10년차방탄팬인데 우리방탄처럼 성공은못하겠지만 일단 방탄의 부하가되고싶다는거니 이름기억은해둠ㅇㅇ [SEP]']"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 76
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Xk-1Y3_vAbCD",
        "outputId": "10c9bdd3-64f1-4fa1-e612-0c3e201af371"
      },
      "source": [
        "train['label'] =train['label'].replace({\"none\":0,\"offensive\":1,\"hate\":2})\n",
        "labels=train['label'].values\n",
        "labels"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([2, 0, 2, ..., 0, 0, 0])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 77
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 70
        },
        "id": "Bi1FUsKrq_Qo",
        "outputId": "e63907de-46a6-4051-b02f-5499a4ebabd3"
      },
      "source": [
        "'''\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "\n",
        "encoder = LabelEncoder()\n",
        "encoder.fit(train['label'])\n",
        "train['label'] = encoder.fit_transform(train['label'])\n",
        "labels=train['label'].values\n",
        "#train.info()\n",
        "labels\n",
        "#hate:0, none:1, offensive:2\n",
        "'''"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "\"\\nfrom sklearn.preprocessing import LabelEncoder\\n\\nencoder = LabelEncoder()\\nencoder.fit(train['label'])\\ntrain['label'] = encoder.fit_transform(train['label'])\\nlabels=train['label'].values\\n#train.info()\\nlabels\\n#hate:0, none:1, offensive:2\\n\""
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 78
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lCD4-WE-Be_y",
        "outputId": "6366f42a-353f-44e6-ce83-c49be3cfcf63"
      },
      "source": [
        "train['label']"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0       2\n",
              "1       0\n",
              "2       2\n",
              "3       0\n",
              "4       2\n",
              "       ..\n",
              "7891    0\n",
              "7892    0\n",
              "7893    0\n",
              "7894    0\n",
              "7895    0\n",
              "Name: label, Length: 7896, dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 79
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uOcVW75_rCG3",
        "outputId": "2c1bd24b-d771-4bd8-e157-b46b9b51b446"
      },
      "source": [
        "# 라벨 추출\n",
        "labels = train['label'].values\n",
        "labels"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([2, 0, 2, ..., 0, 0, 0])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 80
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "T15P0ypGrFqi",
        "outputId": "397bfd88-43b1-4c41-be52-42f714448f30"
      },
      "source": [
        "# BERT의 토크나이저로 문장을 토큰으로 분리\n",
        "tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)\n",
        "tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]\n",
        "\n",
        "print (sentences[0])\n",
        "print (tokenized_texts[0])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[CLS] (현재 호텔주인 심정) 아18 난 마른하늘에 날벼락맞고 호텔망하게생겼는데 누군 계속 추모받네.... [SEP]\n",
            "['[CLS]', '(', '현재', '호', '##텔', '##주', '##인', '심', '##정', ')', '아', '##18', '난', '마', '##른', '##하', '##늘', '##에', '날', '##벼', '##락', '##맞', '##고', '호', '##텔', '##망', '##하게', '##생', '##겼', '##는데', '누', '##군', '계속', '추', '##모', '##받', '##네', '.', '.', '.', '.', '[SEP]']\n"
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
        "id": "p-Uax8-UrJRQ",
        "outputId": "91bcf11d-17a6-4657-e102-84b8020c60a3"
      },
      "source": [
        "# 입력 토큰의 최대 시퀀스 길이\n",
        "MAX_LEN = 128\n",
        "\n",
        "# 토큰을 숫자 인덱스로 변환\n",
        "input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]\n",
        "\n",
        "# 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움\n",
        "input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype=\"long\", truncating=\"post\", padding=\"post\")\n",
        "\n",
        "input_ids[0]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([   101,    113,  26565,   9985, 119354,  16323,  12030,   9491,\n",
              "        16605,    114,   9519,  45987,   8984,   9246,  37819,  35506,\n",
              "       118762,  10530,   8985, 118983, 107693, 118913,  11664,   9985,\n",
              "       119354,  89292,  17594,  24017, 118637,  41850,   9032,  17360,\n",
              "        77039,   9765,  39420, 118965,  77884,    119,    119,    119,\n",
              "          119,    102,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 82
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Xt4rP9SorL6E",
        "outputId": "307a3be4-b279-44ed-8a1c-b59cbe4c67c8"
      },
      "source": [
        "# 어텐션 마스크 초기화\n",
        "attention_masks = []\n",
        "\n",
        "# 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정\n",
        "# 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상\n",
        "for seq in input_ids:\n",
        "    seq_mask = [float(i>0) for i in seq]\n",
        "    attention_masks.append(seq_mask)\n",
        "\n",
        "print(attention_masks[0])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
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
        "id": "8xmc8F_XrOx-",
        "outputId": "f571cf4a-0f2a-42c6-a226-ef1f93f24d5a"
      },
      "source": [
        "#train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, train['label'].values, random_state=42, test_size=0.1)\n",
        "train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=42, test_size=0.1)\n",
        "train_masks, validation_masks, _, _ = train_test_split(attention_masks, \n",
        "                                                       input_ids,\n",
        "                                                       random_state=42, \n",
        "                                                       test_size=0.1)\n",
        "\n",
        "\n",
        "\n",
        "# 데이터를 파이토치의 텐서로 변환\n",
        "train_inputs = torch.tensor(train_inputs)\n",
        "train_labels = torch.tensor(train_labels)\n",
        "train_masks = torch.tensor(train_masks)\n",
        "validation_inputs = torch.tensor(validation_inputs)\n",
        "validation_labels = torch.tensor(validation_labels)\n",
        "validation_masks = torch.tensor(validation_masks)\t\t\t\t\n",
        "\n",
        "print(train_inputs[0])\n",
        "print(train_labels[0])\n",
        "print(train_masks[0])\n",
        "print(validation_inputs[0])\n",
        "print(validation_labels[0])\n",
        "print(validation_masks[0])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "tensor([   101,   8853,  11903,  25503,  96720, 110176,  26344,  28911,    132,\n",
            "           132,    132,    102,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0])\n",
            "tensor(1)\n",
            "tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0.])\n",
            "tensor([   101,   9672,  51431,    198,    198,   9365, 118702,  30873,  21614,\n",
            "        119219, 119113, 119118,  32158,    119,    119,   9638, 119439,  10892,\n",
            "        119164,  35506, 119050,  16439,  48549,    136,    136,   9405,  24017,\n",
            "        119446,  12508,  48387,  37712,  11102, 119060,  20309,  34994,  11287,\n",
            "          9895,  29455,  16439,  12638,  12424,  14843,  22333,  27654,  16985,\n",
            "        118767, 119219,    119,    119,    102,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0])\n",
            "tensor(2)\n",
            "tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,\n",
            "        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,\n",
            "        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0.])\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JLW2Tw6arfQ2"
      },
      "source": [
        "# 배치 사이즈\n",
        "batch_size = 32\n",
        "\n",
        "# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정\n",
        "# 학습시 배치 사이즈 만큼 데이터를 가져옴\n",
        "train_data = TensorDataset(train_inputs, train_masks, train_labels)\n",
        "train_sampler = RandomSampler(train_data)\n",
        "train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)\n",
        "\n",
        "validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)\n",
        "validation_sampler = SequentialSampler(validation_data)\n",
        "validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fl3Dj47ormxd"
      },
      "source": [
        "# 전처리-테스트셋"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VKHaRllMrg9V",
        "outputId": "4c4fa091-6133-402e-da5d-d3b66fa1de96"
      },
      "source": [
        "# 리뷰 문장 추출\n",
        "sentences = dev['comments']\n",
        "sentences[:10]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0                          송중기 시대극은 믿고본다. 첫회 신선하고 좋았다.\n",
              "1                                              지현우 나쁜놈\n",
              "2           알바쓰고많이만들면되지 돈욕심없으면골목식당왜나온겨 기댕기게나하고 산에가서팔어라\n",
              "3                                     설마 ㅈ 현정 작가 아니지??\n",
              "4    이미자씨 송혜교씨 돈이 그리 많으면 탈세말고 그돈으로 평소에 불우이웃에게 기부도 좀...\n",
              "5                                     일베충들 ㅂㄷ거리는것봐라 ㅉㅉ\n",
              "6    아이즈원 힘내세요...일본 진출도 했으니 일본서 좋은 모습 보여줘도 팬들은 응원 합니다.\n",
              "7                강부자 선생님 전미선 비보에 오열을 하셨다니 눈물이 나네요 힘내세요\n",
              "8                                               알았어 그만\n",
              "9    이영자씨는 진정성 있는거라면 녹화불참으로 끝내지말고 자진하차해라 시청자는 고려도 안...\n",
              "Name: comments, dtype: object"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 86
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NtX4QTeOqP80",
        "outputId": "c6d1ae5e-f0ce-4623-8db4-99070f618691"
      },
      "source": [
        "# BERT의 입력 형식에 맞게 변환\n",
        "sentences = [\"[CLS] \" + str(sentence) + \" [SEP]\" for sentence in sentences]\n",
        "sentences[:10]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['[CLS] 송중기 시대극은 믿고본다. 첫회 신선하고 좋았다. [SEP]',\n",
              " '[CLS] 지현우 나쁜놈 [SEP]',\n",
              " '[CLS] 알바쓰고많이만들면되지 돈욕심없으면골목식당왜나온겨 기댕기게나하고 산에가서팔어라 [SEP]',\n",
              " '[CLS] 설마 ㅈ 현정 작가 아니지?? [SEP]',\n",
              " '[CLS] 이미자씨 송혜교씨 돈이 그리 많으면 탈세말고 그돈으로 평소에 불우이웃에게 기부도 좀 하고사시죠. [SEP]',\n",
              " '[CLS] 일베충들 ㅂㄷ거리는것봐라 ㅉㅉ [SEP]',\n",
              " '[CLS] 아이즈원 힘내세요...일본 진출도 했으니 일본서 좋은 모습 보여줘도 팬들은 응원 합니다. [SEP]',\n",
              " '[CLS] 강부자 선생님 전미선 비보에 오열을 하셨다니 눈물이 나네요 힘내세요 [SEP]',\n",
              " '[CLS] 알았어 그만 [SEP]',\n",
              " '[CLS] 이영자씨는 진정성 있는거라면 녹화불참으로 끝내지말고 자진하차해라 시청자는 고려도 안하고 일방적 불참은 아닌듯 엠비씨도 시청율 좋아서 고민하는거 같은데 결방할게 아니고 폐지해라 [SEP]']"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 87
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WQPqR_yUrvCc",
        "outputId": "78389377-1893-4d3b-f231-805d2cdab302"
      },
      "source": [
        "dev['label'] =dev['label'].replace({\"none\":0,\"offensive\":1,\"hate\":2})\n",
        "labels=dev['label'].values\n",
        "labels"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 1, 2, 2, 1, 2, 0, 0, 1, 1, 0, 2, 0, 1, 1, 1, 2, 2, 2, 1, 1, 0,\n",
              "       1, 2, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 1,\n",
              "       0, 1, 2, 2, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 2, 0,\n",
              "       1, 1, 0, 2, 1, 1, 0, 2, 1, 0, 1, 2, 1, 1, 0, 1, 2, 1, 0, 0, 0, 0,\n",
              "       1, 1, 0, 0, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 0, 2, 1, 2, 2, 1, 0, 1,\n",
              "       0, 1, 2, 1, 1, 1, 0, 0, 2, 2, 1, 0, 2, 0, 1, 0, 0, 2, 2, 0, 0, 0,\n",
              "       2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0, 2, 0, 2, 1, 2, 0, 1,\n",
              "       1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 2, 1, 1, 1, 2, 2,\n",
              "       1, 0, 0, 2, 0, 2, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 2, 1, 2, 1,\n",
              "       1, 1, 0, 2, 0, 2, 1, 0, 0, 2, 0, 1, 2, 2, 2, 2, 0, 2, 1, 1, 2, 2,\n",
              "       1, 0, 0, 2, 2, 2, 1, 0, 0, 1, 1, 2, 1, 0, 0, 1, 0, 1, 1, 1, 0, 2,\n",
              "       1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 2, 0, 2, 1, 0, 0, 1, 1, 0, 0, 1, 2,\n",
              "       2, 0, 0, 1, 1, 2, 1, 2, 1, 2, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1,\n",
              "       2, 0, 0, 2, 0, 2, 2, 1, 0, 1, 0, 0, 2, 1, 0, 2, 2, 1, 1, 0, 0, 1,\n",
              "       1, 1, 2, 0, 1, 0, 2, 0, 1, 1, 0, 2, 2, 2, 2, 1, 1, 1, 0, 2, 2, 2,\n",
              "       1, 2, 1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 2, 1, 0, 0, 1, 2, 1, 0, 1, 2,\n",
              "       1, 1, 1, 0, 0, 1, 1, 2, 0, 1, 2, 1, 1, 1, 1, 0, 2, 0, 2, 0, 1, 1,\n",
              "       1, 0, 1, 0, 2, 1, 1, 2, 0, 0, 1, 0, 0, 2, 1, 2, 2, 2, 0, 1, 1, 2,\n",
              "       1, 0, 0, 1, 1, 1, 0, 2, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 0, 0,\n",
              "       1, 0, 0, 0, 0, 1, 0, 1, 2, 0, 1, 1, 1, 0, 2, 1, 2, 1, 1, 1, 1, 1,\n",
              "       2, 0, 1, 1, 0, 2, 1, 1, 2, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 0, 2, 0,\n",
              "       2, 1, 1, 0, 1, 2, 1, 2, 0])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 88
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sJ5oUTUyrydz",
        "outputId": "8f17d598-2b31-4dc2-983e-b0053a387b29"
      },
      "source": [
        "# 라벨 추출\n",
        "labels = dev['label'].values\n",
        "labels"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 1, 2, 2, 1, 2, 0, 0, 1, 1, 0, 2, 0, 1, 1, 1, 2, 2, 2, 1, 1, 0,\n",
              "       1, 2, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 1,\n",
              "       0, 1, 2, 2, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 2, 0,\n",
              "       1, 1, 0, 2, 1, 1, 0, 2, 1, 0, 1, 2, 1, 1, 0, 1, 2, 1, 0, 0, 0, 0,\n",
              "       1, 1, 0, 0, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 0, 2, 1, 2, 2, 1, 0, 1,\n",
              "       0, 1, 2, 1, 1, 1, 0, 0, 2, 2, 1, 0, 2, 0, 1, 0, 0, 2, 2, 0, 0, 0,\n",
              "       2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0, 2, 0, 2, 1, 2, 0, 1,\n",
              "       1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 2, 1, 1, 1, 2, 2,\n",
              "       1, 0, 0, 2, 0, 2, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 2, 1, 2, 1,\n",
              "       1, 1, 0, 2, 0, 2, 1, 0, 0, 2, 0, 1, 2, 2, 2, 2, 0, 2, 1, 1, 2, 2,\n",
              "       1, 0, 0, 2, 2, 2, 1, 0, 0, 1, 1, 2, 1, 0, 0, 1, 0, 1, 1, 1, 0, 2,\n",
              "       1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 2, 0, 2, 1, 0, 0, 1, 1, 0, 0, 1, 2,\n",
              "       2, 0, 0, 1, 1, 2, 1, 2, 1, 2, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1,\n",
              "       2, 0, 0, 2, 0, 2, 2, 1, 0, 1, 0, 0, 2, 1, 0, 2, 2, 1, 1, 0, 0, 1,\n",
              "       1, 1, 2, 0, 1, 0, 2, 0, 1, 1, 0, 2, 2, 2, 2, 1, 1, 1, 0, 2, 2, 2,\n",
              "       1, 2, 1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 2, 1, 0, 0, 1, 2, 1, 0, 1, 2,\n",
              "       1, 1, 1, 0, 0, 1, 1, 2, 0, 1, 2, 1, 1, 1, 1, 0, 2, 0, 2, 0, 1, 1,\n",
              "       1, 0, 1, 0, 2, 1, 1, 2, 0, 0, 1, 0, 0, 2, 1, 2, 2, 2, 0, 1, 1, 2,\n",
              "       1, 0, 0, 1, 1, 1, 0, 2, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 0, 0,\n",
              "       1, 0, 0, 0, 0, 1, 0, 1, 2, 0, 1, 1, 1, 0, 2, 1, 2, 1, 1, 1, 1, 1,\n",
              "       2, 0, 1, 1, 0, 2, 1, 1, 2, 0, 0, 0, 0, 2, 0, 0, 1, 1, 1, 0, 2, 0,\n",
              "       2, 1, 1, 0, 1, 2, 1, 2, 0])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 89
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "67h28o82r1U0",
        "outputId": "2b65d0fc-4b5a-401e-c9ae-6e100e5530c1"
      },
      "source": [
        "# BERT의 토크나이저로 문장을 토큰으로 분리\n",
        "tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)\n",
        "tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]\n",
        "\n",
        "print (sentences[0])\n",
        "print (tokenized_texts[0])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[CLS] 송중기 시대극은 믿고본다. 첫회 신선하고 좋았다. [SEP]\n",
            "['[CLS]', '송', '##중', '##기', '시대', '##극', '##은', '믿', '##고', '##본', '##다', '.', '첫', '##회', '신', '##선', '##하고', '좋', '##았다', '.', '[SEP]']\n"
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
        "id": "U2UTQK4xr4Dd",
        "outputId": "6b796b13-a723-400b-8382-efd61c6e27be"
      },
      "source": [
        "# 입력 토큰의 최대 시퀀스 길이\n",
        "MAX_LEN = 128\n",
        "\n",
        "# 토큰을 숫자 인덱스로 변환\n",
        "input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]\n",
        "\n",
        "# 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움\n",
        "input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype=\"long\", truncating=\"post\", padding=\"post\")\n",
        "\n",
        "input_ids[0]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([   101,   9454,  41693,  12310, 102080,  63243,  10892,   9312,\n",
              "        11664,  40419,  11903,    119,   9750,  14863,   9487,  18471,\n",
              "        12453,   9685,  27303,    119,    102,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0,\n",
              "            0,      0,      0,      0,      0,      0,      0,      0])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 91
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "I0rTT-kAr7Yj",
        "outputId": "b24763c3-d9f7-4117-ae0e-1dc5fe0058ab"
      },
      "source": [
        "# 어텐션 마스크 초기화\n",
        "attention_masks = []\n",
        "\n",
        "# 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정\n",
        "# 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상\n",
        "for seq in input_ids:\n",
        "    seq_mask = [float(i>0) for i in seq]\n",
        "    attention_masks.append(seq_mask)\n",
        "\n",
        "print(attention_masks[0])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n"
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
        "id": "KubhwRf7r9_s",
        "outputId": "4e9631ce-7f51-4861-939c-d391645167ca"
      },
      "source": [
        "# 데이터를 파이토치의 텐서로 변환\n",
        "test_inputs = torch.tensor(input_ids)\n",
        "test_labels = torch.tensor(labels)\n",
        "test_masks = torch.tensor(attention_masks)\n",
        "\n",
        "print(test_inputs[0])\n",
        "print(test_labels[0])\n",
        "print(test_masks[0])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "tensor([   101,   9454,  41693,  12310, 102080,  63243,  10892,   9312,  11664,\n",
            "         40419,  11903,    119,   9750,  14863,   9487,  18471,  12453,   9685,\n",
            "         27303,    119,    102,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0,      0,      0,      0,      0,      0,      0,      0,\n",
            "             0,      0])\n",
            "tensor(0)\n",
            "tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,\n",
            "        1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
            "        0., 0.])\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uCdiGj1_sncP"
      },
      "source": [
        "# 배치 사이즈\n",
        "batch_size = 32\n",
        "\n",
        "# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정\n",
        "# 학습시 배치 사이즈 만큼 데이터를 가져옴\n",
        "test_data = TensorDataset(test_inputs, test_masks, test_labels)\n",
        "test_sampler = RandomSampler(test_data)\n",
        "test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Lp6e73UsssI9"
      },
      "source": [
        "#모델생성"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JQUBRG25su21",
        "outputId": "8410d7ed-67e3-4558-fb49-cbc4a3d7056c"
      },
      "source": [
        "# GPU 디바이스 이름 구함\n",
        "device_name = tf.test.gpu_device_name()\n",
        "\n",
        "# GPU 디바이스 이름 검사\n",
        "if device_name == '/device:GPU:0':\n",
        "    print('Found GPU at: {}'.format(device_name))\n",
        "else:\n",
        "    raise SystemError('GPU device not found')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found GPU at: /device:GPU:0\n"
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
        "id": "YYlAmRhos256",
        "outputId": "d55baa6d-fe6e-4ac8-9e5d-e73bb9ff641c"
      },
      "source": [
        "# 디바이스 설정\n",
        "if torch.cuda.is_available():    \n",
        "    device = torch.device(\"cuda\")\n",
        "    print('There are %d GPU(s) available.' % torch.cuda.device_count())\n",
        "    print('We will use the GPU:', torch.cuda.get_device_name(0))\n",
        "else:\n",
        "    device = torch.device(\"cpu\")\n",
        "    print('No GPU available, using the CPU instead.')"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "There are 1 GPU(s) available.\n",
            "We will use the GPU: Tesla T4\n"
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
        "id": "X-X_OqsYs5_r",
        "outputId": "9a2abe73-960d-4298-a5b2-2bf4c40fddfa"
      },
      "source": [
        "# 분류를 위한 BERT 모델 생성\n",
        "model = BertForSequenceClassification.from_pretrained(\"bert-base-multilingual-cased\", num_labels=3)\n",
        "model.cuda()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Some weights of the model checkpoint at bert-base-multilingual-cased were not used when initializing BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']\n",
            "- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n",
            "- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).\n",
            "Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-multilingual-cased and are newly initialized: ['classifier.weight', 'classifier.bias']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "BertForSequenceClassification(\n",
              "  (bert): BertModel(\n",
              "    (embeddings): BertEmbeddings(\n",
              "      (word_embeddings): Embedding(119547, 768, padding_idx=0)\n",
              "      (position_embeddings): Embedding(512, 768)\n",
              "      (token_type_embeddings): Embedding(2, 768)\n",
              "      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "      (dropout): Dropout(p=0.1, inplace=False)\n",
              "    )\n",
              "    (encoder): BertEncoder(\n",
              "      (layer): ModuleList(\n",
              "        (0): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (1): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (2): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (3): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (4): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (5): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (6): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (7): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (8): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (9): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (10): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "        (11): BertLayer(\n",
              "          (attention): BertAttention(\n",
              "            (self): BertSelfAttention(\n",
              "              (query): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (key): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (value): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "            (output): BertSelfOutput(\n",
              "              (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "              (dropout): Dropout(p=0.1, inplace=False)\n",
              "            )\n",
              "          )\n",
              "          (intermediate): BertIntermediate(\n",
              "            (dense): Linear(in_features=768, out_features=3072, bias=True)\n",
              "          )\n",
              "          (output): BertOutput(\n",
              "            (dense): Linear(in_features=3072, out_features=768, bias=True)\n",
              "            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n",
              "            (dropout): Dropout(p=0.1, inplace=False)\n",
              "          )\n",
              "        )\n",
              "      )\n",
              "    )\n",
              "    (pooler): BertPooler(\n",
              "      (dense): Linear(in_features=768, out_features=768, bias=True)\n",
              "      (activation): Tanh()\n",
              "    )\n",
              "  )\n",
              "  (dropout): Dropout(p=0.1, inplace=False)\n",
              "  (classifier): Linear(in_features=768, out_features=3, bias=True)\n",
              ")"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 97
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BSOi7YwjtGsp"
      },
      "source": [
        "# 옵티마이저 설정\n",
        "optimizer = AdamW(model.parameters(),\n",
        "                  lr = 2e-5, # 학습률\n",
        "                  eps = 1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값\n",
        "                )\n",
        "\n",
        "# 에폭수\n",
        "epochs = 4\n",
        "\n",
        "# 총 훈련 스텝 : 배치반복 횟수 * 에폭\n",
        "total_steps = len(train_dataloader) * epochs\n",
        "\n",
        "# 처음에 학습률을 조금씩 변화시키는 스케줄러 생성\n",
        "scheduler = get_linear_schedule_with_warmup(optimizer, \n",
        "                                            num_warmup_steps = 0,\n",
        "                                            num_training_steps = total_steps)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Qk5UI14It5Mm"
      },
      "source": [
        "# 모델학습"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RFMNL6Jtt9A1"
      },
      "source": [
        "# 정확도 계산 함수\n",
        "def flat_accuracy(preds, labels):\n",
        "    \n",
        "    pred_flat = np.argmax(preds, axis=1).flatten()\n",
        "    labels_flat = labels.flatten()\n",
        "\n",
        "    return np.sum(pred_flat == labels_flat) / len(labels_flat)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "grdSr_xNuBAX"
      },
      "source": [
        "# 시간 표시 함수\n",
        "def format_time(elapsed):\n",
        "\n",
        "    # 반올림\n",
        "    elapsed_rounded = int(round((elapsed)))\n",
        "    \n",
        "    # hh:mm:ss으로 형태 변경\n",
        "    return str(datetime.timedelta(seconds=elapsed_rounded))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4ksEdXB7uDY1",
        "outputId": "2084530d-dc99-4c2e-9ba8-d1fef3dd795d"
      },
      "source": [
        "# 재현을 위해 랜덤시드 고정\n",
        "seed_val = 42\n",
        "random.seed(seed_val)\n",
        "np.random.seed(seed_val)\n",
        "torch.manual_seed(seed_val)\n",
        "torch.cuda.manual_seed_all(seed_val)\n",
        "\n",
        "# 그래디언트 초기화\n",
        "model.zero_grad()\n",
        "\n",
        "# 에폭만큼 반복\n",
        "for epoch_i in range(0, epochs):\n",
        "    \n",
        "    # ========================================\n",
        "    #               Training\n",
        "    # ========================================\n",
        "    \n",
        "    print(\"\")\n",
        "    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))\n",
        "    print('Training...')\n",
        "\n",
        "    # 시작 시간 설정\n",
        "    t0 = time.time()\n",
        "\n",
        "    # 로스 초기화\n",
        "    total_loss = 0\n",
        "\n",
        "    # 훈련모드로 변경\n",
        "    model.train()\n",
        "        \n",
        "    # 데이터로더에서 배치만큼 반복하여 가져옴\n",
        "    for step, batch in enumerate(train_dataloader):\n",
        "        # 경과 정보 표시\n",
        "        if step % 500 == 0 and not step == 0:\n",
        "            elapsed = format_time(time.time() - t0)\n",
        "            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))\n",
        "\n",
        "        # 배치를 GPU에 넣음\n",
        "        batch = tuple(t.to(device) for t in batch)\n",
        "        \n",
        "        # 배치에서 데이터 추출\n",
        "        b_input_ids, b_input_mask, b_labels = batch\n",
        "\n",
        "        # Forward 수행                \n",
        "        outputs = model(b_input_ids, \n",
        "                        token_type_ids=None, \n",
        "                        attention_mask=b_input_mask, \n",
        "                        labels=b_labels)\n",
        "        \n",
        "        # 로스 구함\n",
        "        loss = outputs[0]\n",
        "\n",
        "        # 총 로스 계산\n",
        "        total_loss += loss.item()\n",
        "\n",
        "        # Backward 수행으로 그래디언트 계산\n",
        "        loss.backward()\n",
        "\n",
        "        # 그래디언트 클리핑\n",
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
        "\n",
        "        # 그래디언트를 통해 가중치 파라미터 업데이트\n",
        "        optimizer.step()\n",
        "\n",
        "        # 스케줄러로 학습률 감소\n",
        "        scheduler.step()\n",
        "\n",
        "        # 그래디언트 초기화\n",
        "        model.zero_grad()\n",
        "\n",
        "    # 평균 로스 계산\n",
        "    avg_train_loss = total_loss / len(train_dataloader)            \n",
        "\n",
        "    print(\"\")\n",
        "    print(\"  Average training loss: {0:.2f}\".format(avg_train_loss))\n",
        "    print(\"  Training epcoh took: {:}\".format(format_time(time.time() - t0)))\n",
        "        \n",
        "    # ========================================\n",
        "    #               Validation\n",
        "    # ========================================\n",
        "\n",
        "    print(\"\")\n",
        "    print(\"Running Validation...\")\n",
        "\n",
        "    #시작 시간 설정\n",
        "    t0 = time.time()\n",
        "\n",
        "    # 평가모드로 변경\n",
        "    model.eval()\n",
        "\n",
        "    # 변수 초기화\n",
        "    eval_loss, eval_accuracy = 0, 0\n",
        "    nb_eval_steps, nb_eval_examples = 0, 0\n",
        "\n",
        "    # 데이터로더에서 배치만큼 반복하여 가져옴\n",
        "    for batch in validation_dataloader:\n",
        "        # 배치를 GPU에 넣음\n",
        "        batch = tuple(t.to(device) for t in batch)\n",
        "        \n",
        "        # 배치에서 데이터 추출\n",
        "        b_input_ids, b_input_mask, b_labels = batch\n",
        "        \n",
        "        # 그래디언트 계산 안함\n",
        "        with torch.no_grad():     \n",
        "            # Forward 수행\n",
        "            outputs = model(b_input_ids, \n",
        "                            token_type_ids=None, \n",
        "                            attention_mask=b_input_mask)\n",
        "        \n",
        "        # 로스 구함\n",
        "        logits = outputs[0]\n",
        "        \n",
        "        # CPU로 데이터 이동\n",
        "        logits = logits.detach().cpu().numpy()\n",
        "        label_ids = b_labels.to('cpu').numpy()\n",
        "        \n",
        "        # 출력 로짓과 라벨을 비교하여 정확도 계산\n",
        "        tmp_eval_accuracy = flat_accuracy(logits, label_ids)\n",
        "        eval_accuracy += tmp_eval_accuracy\n",
        "        nb_eval_steps += 1\n",
        "\n",
        "    print(\"  Accuracy: {0:.2f}\".format(eval_accuracy/nb_eval_steps))\n",
        "    print(\"  Validation took: {:}\".format(format_time(time.time() - t0)))\n",
        "\n",
        "print(\"\")\n",
        "print(\"Training complete!\")"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\n",
            "======== Epoch 1 / 4 ========\n",
            "Training...\n",
            "\n",
            "  Average training loss: 0.49\n",
            "  Training epcoh took: 0:02:31\n",
            "\n",
            "Running Validation...\n",
            "  Accuracy: 0.60\n",
            "  Validation took: 0:00:06\n",
            "\n",
            "======== Epoch 2 / 4 ========\n",
            "Training...\n",
            "\n",
            "  Average training loss: 0.43\n",
            "  Training epcoh took: 0:02:31\n",
            "\n",
            "Running Validation...\n",
            "  Accuracy: 0.60\n",
            "  Validation took: 0:00:06\n",
            "\n",
            "======== Epoch 3 / 4 ========\n",
            "Training...\n",
            "\n",
            "  Average training loss: 0.44\n",
            "  Training epcoh took: 0:02:31\n",
            "\n",
            "Running Validation...\n",
            "  Accuracy: 0.60\n",
            "  Validation took: 0:00:06\n",
            "\n",
            "======== Epoch 4 / 4 ========\n",
            "Training...\n",
            "\n",
            "  Average training loss: 0.51\n",
            "  Training epcoh took: 0:02:31\n",
            "\n",
            "Running Validation...\n",
            "  Accuracy: 0.60\n",
            "  Validation took: 0:00:06\n",
            "\n",
            "Training complete!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bM2qM_4mxCE1"
      },
      "source": [
        "# 테스트 셋 평가"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Y2YolpjhxHcr",
        "outputId": "d15beed5-e66d-4acd-df47-04eee62f39e9"
      },
      "source": [
        "\n",
        "#시작 시간 설정\n",
        "t0 = time.time()\n",
        "\n",
        "# 평가모드로 변경\n",
        "model.eval()\n",
        "\n",
        "# 변수 초기화\n",
        "eval_loss, eval_accuracy = 0, 0\n",
        "nb_eval_steps, nb_eval_examples = 0, 0\n",
        "\n",
        "# 데이터로더에서 배치만큼 반복하여 가져옴\n",
        "for step, batch in enumerate(test_dataloader):\n",
        "    # 경과 정보 표시\n",
        "    if step % 100 == 0 and not step == 0:\n",
        "        elapsed = format_time(time.time() - t0)\n",
        "        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))\n",
        "\n",
        "    # 배치를 GPU에 넣음\n",
        "    batch = tuple(t.to(device) for t in batch)\n",
        "    \n",
        "    # 배치에서 데이터 추출\n",
        "    b_input_ids, b_input_mask, b_labels = batch\n",
        "    \n",
        "    # 그래디언트 계산 안함\n",
        "    with torch.no_grad():     \n",
        "        # Forward 수행\n",
        "        outputs = model(b_input_ids, \n",
        "                        token_type_ids=None, \n",
        "                        attention_mask=b_input_mask)\n",
        "    \n",
        "    # 로스 구함\n",
        "    logits = outputs[0]\n",
        "\n",
        "    # CPU로 데이터 이동\n",
        "    logits = logits.detach().cpu().numpy()\n",
        "    label_ids = b_labels.to('cpu').numpy()\n",
        " \n",
        "\n",
        "    # 출력 로짓과 라벨을 비교하여 정확도 계산\n",
        "    tmp_eval_accuracy = flat_accuracy(logits, label_ids)\n",
        "    eval_accuracy += tmp_eval_accuracy\n",
        "    nb_eval_steps += 1\n",
        "\n",
        "print(\"\")\n",
        "print(\"Accuracy: {0:.2f}\".format(eval_accuracy/nb_eval_steps))\n",
        "print(\"Test took: {:}\".format(format_time(time.time() - t0)))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Accuracy: 0.60\n",
            "Test took: 0:00:03\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lI30hrw6xQXc"
      },
      "source": [
        "# 새로운 문장 테스트"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Onh5cs7Dyut9",
        "outputId": "bc834699-704e-472e-bf75-eceaa72ec187"
      },
      "source": [
        "# 리뷰 문장 추출\n",
        "test_sentences = test['comments']\n",
        "test_sentences[:10]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0         ㅋㅋㅋㅋ 그래도 조아해주는 팬들 많아서 좋겠다 ㅠㅠ 니들은 온유가 안만져줌 ㅠㅠ\n",
              "1                                        둘다 넘 좋다~행복하세요\n",
              "2                 근데 만원이하는 현금결제만 하라고 써놓은집 우리나라에 엄청 많은데\n",
              "3                원곡생각하나도 안나고 러블리즈 신곡나온줄!!! 너무 예쁘게 잘봤어요\n",
              "4                                   장현승 얘도 참 이젠 짠하다...\n",
              "5    신선하게 웃긴다ㅋㅋㅋ역시 동엽신~~!! 장소연님은 진짜 조선족인가 착각할정도로 말투...\n",
              "6                                              누군데 얘네?\n",
              "7    하자 인생들 모아다가 방송에 내보내고, 덜 하자가 교정해서 장사 풀리게 해주는 감동...\n",
              "8                       진짜 라디오 스타 노래한거 보세요 홍진영은비비지도 못함\n",
              "9    쑈 하지마라 짜식아!음주 1번은 실수, 2번은 고의, 3번은 인간쓰레기다.슬금슬금 ...\n",
              "Name: comments, dtype: object"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 121
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RWS01CyCxZds"
      },
      "source": [
        "# 입력 데이터 변환\n",
        "def convert_input_data(sentences):\n",
        "\n",
        "    # BERT의 토크나이저로 문장을 토큰으로 분리\n",
        "    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]\n",
        "\n",
        "    # 입력 토큰의 최대 시퀀스 길이\n",
        "    MAX_LEN = 128\n",
        "\n",
        "    # 토큰을 숫자 인덱스로 변환\n",
        "    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]\n",
        "    \n",
        "    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움\n",
        "    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype=\"long\", truncating=\"post\", padding=\"post\")\n",
        "\n",
        "    # 어텐션 마스크 초기화\n",
        "    attention_masks = []\n",
        "\n",
        "    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정\n",
        "    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상\n",
        "    for seq in input_ids:\n",
        "        seq_mask = [float(i>0) for i in seq]\n",
        "        attention_masks.append(seq_mask)\n",
        "\n",
        "    # 데이터를 파이토치의 텐서로 변환\n",
        "    inputs = torch.tensor(input_ids)\n",
        "    masks = torch.tensor(attention_masks)\n",
        "\n",
        "    return inputs, masks"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VAOygaVOxiJE"
      },
      "source": [
        "# 문장 테스트\n",
        "def test_sentences(sentences):\n",
        "\n",
        "    # 평가모드로 변경\n",
        "    model.eval()\n",
        "\n",
        "    # 문장을 입력 데이터로 변환\n",
        "    inputs, masks = convert_input_data(sentences)\n",
        "\n",
        "    # 데이터를 GPU에 넣음\n",
        "    b_input_ids = inputs.to(device)\n",
        "    b_input_mask = masks.to(device)\n",
        "            \n",
        "    # 그래디언트 계산 안함\n",
        "    with torch.no_grad():     \n",
        "        # Forward 수행\n",
        "        outputs = model(b_input_ids, \n",
        "                        token_type_ids=None, \n",
        "                        attention_mask=b_input_mask)\n",
        "\n",
        "    # 로스 구함\n",
        "    logits = outputs[0]\n",
        "\n",
        "    # CPU로 데이터 이동\n",
        "    logits = logits.detach().cpu().numpy()\n",
        "\n",
        "    return logits"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "t0OXLZTa4ppK",
        "outputId": "746133d5-e68f-4a33-f57b-389b492c13ea"
      },
      "source": [
        "logits = test_sentences(['쑈 하지마라 짜식아!음주 1번은 실수, 2번은 고의, 3번은 인간쓰레기다.슬금슬금'])\n",
        "\n",
        "print(logits)\n",
        "print(np.argmax(logits))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[-1.2494057   0.24309215  0.896483  ]]\n",
            "2\n"
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
        "id": "t2i4DbqTbiYc",
        "outputId": "68694116-b800-4170-f9ac-ec1d8a4f058d"
      },
      "source": [
        "logits = test_sentences(test['comments'])\n",
        "print(logits)\n",
        "print(np.argmax(logits))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[ 0.5849384   0.659094   -1.0256827 ]\n",
            " [ 3.054541   -1.2087638  -1.8388913 ]\n",
            " [ 1.1519105   0.21362758 -1.436918  ]\n",
            " ...\n",
            " [ 1.2328675   0.24678694 -1.4500159 ]\n",
            " [-0.04345606  0.5912328  -0.389204  ]\n",
            " [ 2.5495133  -0.40546557 -2.0031798 ]]\n",
            "75\n"
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
        "id": "S66Ov89txtXH",
        "outputId": "3ead8394-c513-4d9f-ecf0-b4b5a851850c"
      },
      "source": [
        "logits = test_sentences(test['comments'])\n",
        "print(logits)\n",
        "\n",
        "\n",
        "print(np.argmax (logits))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[ 0.5849384   0.659094   -1.0256827 ]\n",
            " [ 3.054541   -1.2087638  -1.8388913 ]\n",
            " [ 1.1519105   0.21362758 -1.436918  ]\n",
            " ...\n",
            " [ 1.2328675   0.24678694 -1.4500159 ]\n",
            " [-0.04345606  0.5912328  -0.389204  ]\n",
            " [ 2.5495133  -0.40546557 -2.0031798 ]]\n",
            "75\n"
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
        "id": "reQYvHjtavXD",
        "outputId": "4fedcfe0-33eb-4e8c-d054-0e27c2f4f78b"
      },
      "source": [
        "logits = test_sentences(test['comments'])\n",
        "\n",
        "print(logits)\n",
        "print(np.argmax(logits))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[[ 0.5849384   0.659094   -1.0256827 ]\n",
            " [ 3.054541   -1.2087638  -1.8388913 ]\n",
            " [ 1.1519105   0.21362758 -1.436918  ]\n",
            " ...\n",
            " [ 1.2328675   0.24678694 -1.4500159 ]\n",
            " [-0.04345606  0.5912328  -0.389204  ]\n",
            " [ 2.5495133  -0.40546557 -2.0031798 ]]\n",
            "75\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8skGkTYrQG7w"
      },
      "source": [
        "logits = test_sentences(test['comments']) \n",
        "    #logits = test_sentences(['누군지도모르는데 펜이있어?음......참....재인씨 고생고생해서 자리잡은사람 가기고놀지마세요천벌받는다'])\n",
        "\n",
        "    #print(test_results)\n",
        "test_labels = np.argmax(logits)\n",
        "    #test_labels = [le.inverse_transform(i) for i in test_results] \n",
        "no_label = pd.DataFrame({'comments':test['comments'],\"label\":test_labels})\n",
        "no_label.to_csv(\"/content/케글/NEW_test.hate.no_label.csv\",index=False)\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BvPhFqfsdOTw"
      },
      "source": [
        "logits = test_sentences(test['comments']) \n",
        "    #logits = test_sentences(['누군지도모르는데 펜이있어?음......참....재인씨 고생고생해서 자리잡은사람 가기고놀지마세요천벌받는다'])\n",
        "\n",
        "    #print(test_results)\n",
        "test_labels = np.argmax(logits)\n",
        "    #test_labels = [le.inverse_transform(i) for i in test_results] \n",
        "    \n",
        "    for tag in tags:\n",
        "        noun = tag['tag']\n",
        "        count = tag['count']\n",
        "        open_output_file.write('{} {}\\n'.format(noun, count))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WlPyMF1v_jmI"
      },
      "source": [
        "def get_tags(text, ntags=50):\n",
        "    spliter = Okt()\n",
        "    # konlpy의 Twitter객체\n",
        "    nouns = spliter.nouns(text)\n",
        "    # nouns 함수를 통해서 text에서 명사만 분리/추출\n",
        "    count = Counter(nouns)\n",
        "    # Counter객체를 생성하고 참조변수 nouns할당\n",
        "    return_list = []  # 명사 빈도수 저장할 변수\n",
        "    for n, c in count.most_common(ntags):\n",
        "        temp = {'tag': n, 'count': c}\n",
        "        return_list.append(temp)\n",
        "    # most_common 메소드는 정수를 입력받아 객체 안의 명사중 빈도수\n",
        "    # 큰 명사부터 순서대로 입력받은 정수 갯수만큼 저장되어있는 객체 반환\n",
        "    # 명사와 사용된 갯수를 return_list에 저장합니다.\n",
        "    return return_list"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}