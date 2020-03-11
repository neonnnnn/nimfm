import httpclient, os, streams, tables, strutils, sequtils, parseutils
import zip/zipfiles, random
import nimfm, nimfm/utils

const nUsers = 943
const nItems = 1682
const nRatings = 100_000
const fileurl = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
const nGenres = 19


proc round(x: int): int =
  result = ((x+5) div 10)*10


proc createUserItemDataset(indices: openarray[int]) =
  var X: CSRDataset
  var y: seq[float64]
  loadUserItemRatingFile("ml-100k/u.data", X, y)
  echo("  Number of samples : ", X.nSamples)
  echo("  Number of features: ", X.nFeatures)
  let nTrain = int(8*X.nSamples/10)
  var
    XTr, XTe: CSRDataset
    yTr, yTe: seq[float64]
  
  (XTr, yTr) = shuffle(X, y, indices[0..<nTrain])
  (XTe, yTe) = shuffle(X, y, indices[nTrain..^1])

  dumpSVMLightFile("ml-100k_user_item_all.svm", X, y)
  dumpSVMLightFile("ml-100k_user_item_train.svm", Xtr, yTr)
  dumpSVMLightFile("ml-100k_user_item_test.svm", XTe, yTe)


proc createUserFeatureMatrix(): CSCDataset = 
  var
    ages: array[nUsers, int]
    genders: array[nUsers, string]
    occupations: array[nUsers, string]
    zipcodes: array[nUsers, char]
    ageEncoder = newLabelEncoder[int]()
    genderEncoder = newLabelEncoder[string]()
    occupationEncoder = newLabelEncoder[string]()
    zipcodeEncoder = newLabelEncoder[char]()

  var i = 0
  for line in "ml-100k/u.user".lines:
    let infos = line.split("|")
    ages[i] = round(parseInt(infos[1]))
    genders[i] = infos[2]
    occupations[i] = infos[3]
    zipcodes[i] = infos[4][0]
    i += 1
  ageEncoder.fit(ages)
  genderEncoder.fit(genders)
  occupationEncoder.fit(occupations)
  zipcodeEncoder.fit(zipcodes)

  var nFeaturesUser = 0
  nFeaturesUser += len(ageEncoder.classes)
  nFeaturesUser += len(genderEncoder.classes)
  nFeaturesUser += len(occupationEncoder.classes)
  nFeaturesUser += len(zipcodeEncoder.classes)
  echo("   Number of user features: ", nFeaturesUser)
  var XUser = newSeqWith(nRatings, newSeqWith(nFeaturesUser, 0.0))
  let agesEnc = ageEncoder.transformed(ages)
  let gendersEnc = genderEncoder.transformed(genders)
  let occupationsEnc = occupationEncoder.transformed(occupations)
  let zipcodesEnc = zipcodeEncoder.transformed(zipcodes)
  i = 0
  for line in "ml-100k/u.data".lines:
    if len(line) < 5:
      continue
    let j = parseInt(line.split("\t")[0])
    var offset = 0
    XUser[i][agesEnc[j-1]+offset] = 1.0
    offset += len(ageEncoder.classes)
    XUser[i][gendersEnc[j-1]+offset] = 1.0
    offset += len(genderEncoder.classes)
    XUser[i][occupationsEnc[j-1]+offset] = 1.0
    offset += len(occupationEncoder.classes)
    XUser[i][zipcodesEnc[j-1]+offset] = 1.0
    i += 1
  result = toCSC(XUser)
 

proc createItemFeatureMatrix(): CSCDataset = 
  var i = 0
  var years = newSeqWith(nItems, 0)
  for line in "ml-100k/u.item".lines:
    if len(line) < 5:
      continue
    let infos = line.split("|")
    try:
      years[i] = round(parseInt(infos[2].split("-")[^1]))
    except:
      years[i] = 0
    i += 1

  var yearEncoder = newLabelEncoder[int]()
  yearEncoder.fit(years)
  echo("   Number of item features: ", len(yearEncoder.classes)+nGenres)
  let yearsEnc = yearEncoder.transformed(years)
  let nFeaturesItems = nGenres + len(yearEncoder.classes)

  var Genres = newSeqWith(nItems, newSeqWith(nGenres, 0.0))
  i = 0
  for line in "ml-100k/u.item".lines:
    if len(line) < 5:
      continue
    let infos = line.split("|")
    for j, val in infos[^nGenres..^1]:
      Genres[i][j] = parseFloat(val)
    i += 1

  var XItems = newSeqWith(nRatings, newSeqWith(nFeaturesItems, 0.0))
  i = 0
  for line in "ml-100k/u.data".lines:
    if len(line) < 5:
      continue
    let item = parseInt(line.split("\t")[1])
    XItems[i][yearsEnc[item-1]] = 1
    for jj, val in Genres[item-1]:
      XItems[i][jj+len(yearEncoder.classes)] = val
    i += 1
  result = toCSC(XItems)


proc createUserItemFeatureDataset(indices: openarray[int]) =
  var
    X: CSCDataset
    y: seq[float64]
    XTr, XTe: CSRDataset
    yTr, yTe: seq[float64]

  loadUserItemRatingFile("ml-100k/u.data", X, y)
  echo("  Create user feature matrix...")
  let XUser = createUserFeatureMatrix()
  echo("  Done.")
  
  echo("  Create item feature matrix...")
  let XItem = createItemFeatureMatrix()
  echo("  Done.")
  
  let XAll = toCSR(hstack([X, XUser, XItem]))
  let nTrain = int(8*X.nSamples/10)
  (XTr, yTr) = shuffle(XAll, y, indices[0..<nTrain])
  (XTe, yTe) = shuffle(XAll, y, indices[nTrain..^1])
  dumpSVMLightFile("ml100k_user_item_feature_all.svm", XAll, y)
  dumpSVMLightFile("ml-100k_user_item_feature_train.svm", Xtr, yTr)
  dumpSVMLightFile("ml-100k_user_item_feature_test.svm", Xte, yTe)
  

when isMainModule:
  var client = newHttpClient()
  if not existsFile("ml-100k.svm"):
    if not existsFile("ml-100k.zip"):
      echo("Download ml-100k data...")
      client.downloadFile(fileurl, "ml-100k.zip")
      echo("Done.")

    echo("Unzip...")
    var z: ZipArchive
    if not z.open("ml-100k.zip", fmRead):
      echo "Open ml-100k.zip failed"
      quit(1)
    z.extractAll(".")
    z.close()
    echo("Done.")
    
    randomize(1)
    var indices = toSeq(0..<nRatings)
    shuffle(indices)

    echo("Create user-item rating matrix and transform it to svmlight format...")    
    createUserItemDataset(indices)
    echo("Done.")

    echo("Create dataset for factorization machines...")
    createUserItemFeatureDataset(indices)
    echo("Done.")
    