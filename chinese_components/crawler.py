# -*- encoding: utf-8 -*-
# Author: hushukai

import time
from bs4 import BeautifulSoup
from selenium import webdriver


def get_char_split_info(unicode_list):
    option = webdriver.ChromeOptions()
    option.headless = True
    chrome = webdriver.Chrome(options=option)

    url = "http://..."

    with open("./char_split_table_crawled.txt", "w", encoding="utf-8") as fw:
        for i, ucode in enumerate(unicode_list):
            chinese_char = chr(ucode)
            raw_ucode = hex(ucode).upper()[2:]
            chrome.get(url)
            chrome.find_element_by_xpath(xpath='//td[span="從中文字查詢部件"]/input[@type="radio"]').click()
            input = chrome.find_element_by_id("vInput")
            input.send_keys(chinese_char)
            button = input.find_element_by_xpath(xpath='ancestor::tbody[1]/child::tr[2]//input[@name="vInputSubmit"]')
            button.click()
            time.sleep(10)
            
            components_list = []
            p_nodes = chrome.find_elements_by_xpath(xpath='//img[@src="images/info.gif"]/parent::a')
            for j in range(len(p_nodes)):
                p_nodes[j].click()
                soup = BeautifulSoup(chrome.page_source, 'lxml')
                u_code = soup.find(name="td", text="統一碼").find_next(name="span", attrs={"id":"ccInfoUnicode"}).get_text()
                c_code = soup.find(name="td", text="*部件碼").find_next(name="span", attrs={"id": "ccInfoCcode"}).get_text()
                chrome.back()
                p_nodes = chrome.find_elements_by_xpath(xpath='//img[@src="images/info.gif"]/parent::a')
                
                if c_code == "---":
                    continue
                components_list.append(c_code)
            
            line_str = raw_ucode + "\t" + chinese_char + "\t" + "/".join(components_list)
            fw.write(line_str + "\n")
            print(i, line_str)
            
    chrome.close()


if __name__ == "__main__":
    MISSING_CHARS_FILE = "missing_chars_in_split_table.txt"
    with open(MISSING_CHARS_FILE, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
    unicodes = [int(line.strip(), base=16) for line in lines]
    get_char_split_info(unicode_list=unicodes)

    print("Done !")
