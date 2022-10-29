def convertText(file):
  my_file = open(file, "r")
  data = my_file.read()
  data_into_list = data.split("\n")
  my_file.close()
  return data_into_list


lexicon = []
lexicon = convertText("negative_words.txt")
print(convertText("positive_words.txt"))
#print(lexicon)