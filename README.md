## ��Ŀ����
���� [DiscoDiffusion](https://github.com/alembics/disco-diffusion) �ľ���棬����ѧϰ����չ��
- Ŀǰֻ֧��ͨ�������ı�����ͼƬ�Ĺ��ܣ�ȥ������Ƶ��VR��3D�ȹ��ܡ�
- ͬʱ֧��ͼ�ν��棨GUI����ʽ������Ŀǰ���滹�ܼ򵥣������������Ż���
- ��һ��windows�������򣬽�ѹ��˫��app.exe����������Pyinstaller����ĳ���ᱻɱ���������Ϊ��ľ����������Σ���������Ŀ���֧������Դ���Լ����У���

![screenshot](asset/screenshot.png)

## ���ٿ�ʼ
```bash
# ��װ����
pip install -r requirements.txt
# ����ģ�Ͳ��ŵ�modelsĿ¼��
https://drive.google.com/drive/folders/1nVae7WmWuZx7Syx_sKBBhCxx4TuJ99ls
# ���д���
python discodiff.py
# �����н�����򣨽���window���������ԣ�
python gui.py
```
### ���а�װ��
����[��װ��]()����ѹ��˫��app.exe����������

### �Լ����
```bash
pyinstaller app.spec --noconfirm --clean
```
�����ɺ�����`dist/app.exe`����������

## ��л������Ŀ
- [DiscoDiffusion](https://github.com/alembics/disco-diffusion)
- [QGUI](https://github.com/QPT-Family/QGUI)
