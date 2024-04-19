import os

from unetppSAM_specific import gen_ans_specific

def gen_ans_directory(config, is_show_ans = True, is_gen_compare = True, is_show_compare = True, directory = "sample"):

    for filename in os.listdir(directory):
        file_type = filename[-4:]
        if file_type != ".png":
            break
        not_mask = filename[-9:]
        # print(not_mask)
        if not_mask == "_mask.png":
            continue
        print(filename)
        file_path = directory + '/' + filename
        gen_ans_specific(config, is_show_ans=is_show_ans, is_gen_compare=is_gen_compare, is_show_compare=is_show_compare, image=file_path)

    
if __name__ == "__main__":
    import json
    configfile_path = "json/config.json"
    configfile = open(configfile_path, "r",encoding="utf-8").read()
    config = json.loads(configfile)
    gen_ans_directory(config["oentheSAM.py"])