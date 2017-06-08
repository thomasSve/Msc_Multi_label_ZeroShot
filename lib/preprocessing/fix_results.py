

def fix_file(filename):
    with open(filename + ".txt") as read, open(filename + "fixed.txt","w") as wrt:
        i = 0
        for line in read:
            if i%10000 == 0: print "lines checked: ", i
            line = line.rstrip().split(",")
            processed_line = []
            for item in line: 
                if item.startswith(" ['") and item.endswith("]"):
                    processed_line.append(" "+item[3:-2])
                else:
                    processed_line.append(item)
            line = ",".join(processed_line)
            wrt.write(line + "\n")
            i += 1

if __name__ == "__main__":
    fix_file('results_nus_wide_Test_zs_us_img_lbl_w2v_wiki_300D_ml_yolo_squared_hinge_2_l2_imagenet')
