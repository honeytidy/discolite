# coding:utf-8
from qgui import CreateQGUI
from qgui.banner_tools import GitHub
from qgui.notebook_tools import ChooseDirTextButton, RunButton, RadioObviousToolButton, InputBox
from qgui.notebook_tools import HorizontalToolsCombine, BaseButton
import discodiff
import webbrowser
import os


def infer(args):
    prompt_texts = args['prompt_text'].get()
    device = args['run_mode'].get()
    count = args['count'].get()
    output_dir = args['save_dir'].get()
    discodiff.do_run(prompt_texts=prompt_texts, device=device, output_dir=output_dir, count=count)


def show(args):
    save_dir = args['save_dir'].get()
    webbrowser.open(os.path.realpath(save_dir))


main_gui = CreateQGUI(title="DiscoDiffusion精简版")
main_gui.add_banner_tool(GitHub("https://github.com/honeytidy/discolite"))
main_gui.add_notebook_tool(InputBox(name="prompt_text", default='a beautiful chinese landscape paintings', width=100))
main_gui.add_notebook_tool(InputBox(name="count", default='5', label_info='生成数量', width=3))
main_gui.add_notebook_tool(RadioObviousToolButton(["cpu", "gpu"], name="run_mode", title="运行模式"))
main_gui.add_notebook_tool(ChooseDirTextButton(name="save_dir", label_info="保存位置", entry_info="output"))
buttons = HorizontalToolsCombine([RunButton(infer), BaseButton(bind_func=show, text="打开生成目录")])
main_gui.add_notebook_tool(buttons)

main_gui.set_navigation_info("项目介绍", "DiscoDiffusion生成艺术图片的精简版")

main_gui.run()
