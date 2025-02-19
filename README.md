# NNCreater_forHuman

## 概要
本プログラムはGUI上で自分好みのAIを作成できるアプリです。

本来自分でAIを作成する際にはPytorchやTensorFlowなどの各種ライブラリの構文を覚えて使用することが一般的ですが、このアプリを作成することで特有の構文を覚える必要なくAIの構築が可能です。

## 使用方法
src/main.py を実行するとGUI画面が出てくるので、そこでAIを形作っていくことができます。

具体的には、一番上のModel Name欄には作成するモデルに対して自由に設定した名称を記載してください。そして、左下のAdd Layer欄からモデルに追加するレイヤー(現在選択できるのはLinearとDropout, BatchNormalize)を選択し、真ん中のAdd Activationから活性化関数を選択してください。これを繰り返してモデルが完成したら右下のGenerate Codeボタンを押してください。すると、カレントディレクトリにAIモデルが作成されます。

モデルの学習方法や推論方法はsrc/test_learning.pyを参考にしてください。

(ここではMNISTの手書き数字分類を実装しており、使用するモデルはtest.pyに格納しています。このモデルを自作のものに置き換えて使用する場合にはモデル作成時に、一番最初のLinear層のinput sizeは28*28に設定してください。出力は10です。)

## 環境構築
動作環境にrequirements.txtで必要ライブラリをインストールしてください。
インストール時にエラーが発生する場合があるかと思いますが、その場合はrequirements.txt内の以下の三文を削除してください。

torch==2.6.0+cu124

torchaudio==2.6.0+cu124

torchvision==0.21.0+cu124

削除した場合には、[pytorch公式サイト](https://pytorch.org/get-started/locally/)から自身の環境に適したライブラリを選択してインストールしてください。

## 将来性
現在はMLPと呼ばれる単純な全結合ネットワークしか作成できませんが、ゆくゆくはCNNやRNN,Transformerなどの複雑なモデルもGUI上で構築できるように発展させていきます。

さらには、Skip Connectionなどの構造も導入できるような工夫を施していきたいと思います。

(U-netとかこれで作れたらいいなあ)

## ライセンス

This project is licensed under the GNU LESSER GENERAL PUBLIC LICENSE, see the LICENSE.txt file for details
