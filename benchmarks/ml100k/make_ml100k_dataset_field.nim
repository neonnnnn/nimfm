import httpclient, os, streams, tables, strutils, sequtils, parseutils
import zip/zipfiles, random
import nimfm/dataset, nimfm/utils

const nUsers = 943
const nItems = 1682
const nRatings = 100_000
const fileurl = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
const nGenres = 19


proc round(x: int): int =
  result = ((x+5) div 10)*10


proc createUserFeatureMatrix(): CSRFieldDataset = 
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
  var indices = newSeqWith(nRatings*4, 0)
  var indptr = newSeqWith(nRatings+1, 0)
  var fields = newSeqWith(nRatings*4, 0)
  var data = newSeqWith(nRatings*4, 1.0)
  indptr[0] = 0

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
    indices[i*4] = agesEnc[j-1]
    fields[i*4] = 0
    offset += len(ageEncoder.classes)

    indices[i*4+1] = gendersEnc[j-1]+offset
    fields[i*4+1] = 1
    offset += len(genderEncoder.classes)

    indices[i*4+2] = occupationsEnc[j-1] + offset
    fields[i*4+2] = 2
    offset += len(occupationEncoder.classes)

    indices[i*4+3] = zipcodesEnc[j-1]+offset
    fields[i*4+3] = 3
    i += 1
    indptr[i] = indptr[i-1] + 4
  result = newCSRFieldDataset(data=data, indices=indices, indptr=indptr,
                              fields=fields, nFields=4, nSamples=nRatings,
                              nFeatures=nFeaturesUser)

proc createItemFeatureMatrix(): CSRFieldDataset  = 
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

  var indptr = newSeqWith(nRatings+1, 0)
  var indices = newSeqWith[int](0, 0)
  var fields = newSeqWith[int](0, 0)
  i = 0
  var nnz = 0
  for line in "ml-100k/u.data".lines:
    if len(line) < 5:
      continue
    let item = parseInt(line.split("\t")[1])
    fields.add(0)
    indices.add(yearsEnc[item-1])
    inc(nnz)

    for jj, val in Genres[item-1]:
      if val != 0:
        fields.add(1)
        indices.add(jj+len(yearEncoder.classes))
        inc(nnz)
    inc(i)
    indptr[i] = nnz

  var data = newSeqWith(nnz, 1.0)
  result = newCSRFieldDataset(data=data, indices=indices, indptr=indptr,
                              fields=fields, nFields=2, nSamples=nRatings,
                              nFeatures=nFeaturesItems)


proc createUserItemFeatureDataset(indices: openarray[int]) =
  var
    X: CSRDataset
    y: seq[float64]
    XField: CSRFieldDataset
    XTr, XTe: CSRFieldDataset
    yTr, yTe: seq[float64]

  loadUserItemRatingFile("ml-100k/u.data", X, y)
  var indices = X.data.indices
  var indptr = X.data.indptr
  var data = X.data.data
  var fields = newSeqWith(X.nSamples*2, 0)
  for i in 0..<X.nSamples:
    fields[2*i+1] = 1
  XField = newCSRFieldDataset(data=data, indices=indices, indptr=indptr,
                              fields=fields, nFields=2, nFeatures=X.nFeatures,
                              nSamples=X.nSamples)
  echo("  Create user feature matrix...")
  let XUser = createUserFeatureMatrix()
  echo("  Done.")
  
  echo("  Create item feature matrix...")
  let XItem = createItemFeatureMatrix()
  echo("  Done.")

  echo(XField.nFields, " ", XUser.nFields, " ", XItem.nFields)
  let XAll = hstack(XField, XUser, XItem)
  echo(XAll.nFields)
  let nTrain = int(8*X.nSamples/10)

  var indicesShuffle = toSeq(0..<nRatings)
  (XTr, yTr) = shuffle(XAll, y, indicesShuffle[0..<nTrain])
  (XTe, yTe) = shuffle(XAll, y, indicesShuffle[nTrain..^1])
  dumpFFMFile("dataset/ml-100k_user_item_feature_all.ffm", XAll, y)
  dumpFFMFile("dataset/ml-100k_user_item_feature_train.ffm", XTr, yTr)
  dumpFFMFile("dataset/ml-100k_user_item_feature_test.ffm", Xte, yTe)


when isMainModule:
  var client = newHttpClient()
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

  echo("Create dataset for factorization machines...")
  createUserItemFeatureDataset(indices)
  echo("Done.")
  