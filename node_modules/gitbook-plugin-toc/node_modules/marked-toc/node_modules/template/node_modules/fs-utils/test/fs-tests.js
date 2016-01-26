/**
 * fs-utils <https://github.com/assemble/fs-utils>
 *
 * Copyright (c) 2014 Jon Schlinkert, Brian Woodward, contributors.
 * Licensed under the MIT license.
 */

const expect = require('chai').expect;
const path = require('path');
const file = require('../');
const cwd = process.cwd();

describe('file system methods', function () {

  var testTxtPath = path.join('test', 'fixtures', 'test.txt');
  var testTxtWritePath = path.join('test', 'actual', 'test.txt');
  var testTxtContents = 'FILE CONTENTS!!!';

  var testJsonPath = path.join('test', 'fixtures', 'test.json');
  var testJsonWritePath = path.join('test', 'actual', 'test.json');
  var testJsonContents = {
    "foo": {
      "bar": "baz"
    }
  };

  var testYamlPath = path.join('test', 'fixtures', 'test.yaml');
  var testYamlWritePath = path.join('test', 'actual', 'test.yaml');
  var testYamlContents = {
    "foo": {
      "bar": "baz"
    }
  };

  it('should return true the path exists', function () {
    var expected = true;
    var actual = file.exists(testTxtPath);
    expect(actual).to.eql(expected);
  });

  it('should return false the path does not exist', function () {
    var expected = false;
    var actual = file.exists('.', 'some', 'random', 'file.json');
    expect(actual).to.eql(expected);
  });

  it('should return true if a path is a real file', function() {
    var expected = true;
    var actual = file.isFile('package.json');
    expect(actual).to.eql(expected);

    expected = true;
    actual = file.isFile('README.md');
    expect(actual).to.eql(expected);
  });

  it('should return false if a path is not a real file', function() {
    var expected = false;
    var actual = file.isFile('test');
    expect(actual).to.eql(expected);
  });

  it('should return true if a path is a real directory', function() {
    var expected = true;
    var actual = file.isDir('test');
    expect(actual).to.eql(expected);
  });

  it('should return false if a path is not a real directory', function() {
    var expected = false;
    var actual = file.isDir('package.json');
    expect(actual).to.eql(expected);

    expected = false;
    actual = file.isDir('README.md');
    expect(actual).to.eql(expected);
  });

  it('should read the file', function () {
    var expected = testTxtContents;
    var actual = file.readFileSync(testTxtPath);
    expect(actual).to.eql(expected);
  });

  it('should read the file (async)', function (done) {
    var expected = testTxtContents;
    file.readFile(testTxtPath, function (err, actual) {
      expect(actual).to.eql(expected);
      done();
    });
  });

  it('should read the json file', function () {
    var expected = testJsonContents;
    var actual = file.readJSONSync(testJsonPath);
    expect(actual).to.eql(expected);
  });

  it('should read the json file (async)', function (done) {
    var expected = testJsonContents;
    file.readJSON(testJsonPath, function (err, actual) {
      expect(actual).to.eql(expected);
      done();
    });
  });

  it('should read the yaml file', function () {
    var expected = testYamlContents;
    var actual = file.readYAMLSync(testYamlPath);
    expect(actual).to.eql(expected);
  });

  it('should read the yaml file (async)', function (done) {
    var expected = testYamlContents;
    file.readYAML(testYamlPath, function (err, actual) {
      expect(actual).to.eql(expected);
      done();
    });
  });

  it('should read the json file automatically', function () {
    var expected = testJsonContents;
    var actual = file.readDataSync(testJsonPath);
    expect(actual).to.eql(expected);
  });

  it('should read the json file automatically (async)', function (done) {
    var expected = testJsonContents;
    file.readData(testJsonPath, function (err, actual) {
      expect(actual).to.eql(expected);
      done();
    });
  });

  it('should read the yaml file automatically', function () {
    var expected = testYamlContents;
    var actual = file.readDataSync(testYamlPath);
    expect(actual).to.eql(expected);
  });

  it('should read the yaml file automatically (async)', function (done) {
    var expected = testYamlContents;
    file.readData(testYamlPath, function (err, actual) {
      expect(actual).to.eql(expected);
      done();
    });
  });

  it('should make a directory, asynchronously', function(done) {
    var newDir = ('test', 'actual', 'new', 'folder', 'async');
    file.mkdir(newDir, function(err) {
      if (err) return console.log(err);
      var expected = file.exists(newDir);
      expect(expected).to.be.ok;
      done();
    });
  });

  it('should make a directory, synchronously', function() {
    var newDir = ('test', 'actual', 'new', 'folder', 'sync');
    file.mkdirSync(newDir);
    var expected = file.exists(newDir);
    expect(expected).to.be.ok;
  });

  it('should remove a directory, asynchronously', function(done) {
    var existingDir = ('test', 'actual', 'new', 'folder', 'async');
    file.rmdir(existingDir, function(err) {
      if (err) return console.log(err);
      var expected = !file.exists(existingDir);
      expect(expected).to.be.ok;
      done();
    });
  });

  it('should remove a directory, synchronously', function() {
    var existingDir = ('test', 'actual', 'new', 'folder', 'sync');
    file.rmdirSync(existingDir);
    var expected = !file.exists(existingDir);
    expect(expected).to.be.ok;
  });

  it('should "delete" a directory, synchronously', function() {
    var existingDir = ('test', 'actual', 'new', 'folder', 'sync');
    file.mkdirSync(existingDir);
    file.delete(existingDir);
    var expected = !file.exists(existingDir);
    expect(expected).to.be.ok;
  });

  it('should retrieve file stats, synchronously', function() {
    var stats = file.getStatsSync(testTxtPath);
    expect(stats).to.have.property('mtime');
  });

  it('should retrieve file stats, asynchronously', function(done) {
    file.getStats(testTxtPath, function (err, stats) {
      expect(stats).to.have.property('mtime');
      done();
    });
  });

  it('should throw error when attempting to retrieve stats, synchronously', function() {
    try {
      var stats = file.getStatsSync('some/fake/path/to/fake/file.html');
    } catch (err) {
      expect(err).not.to.be.null;
    }
  });

  it('should return error when attempting to retrieve stats, asynchronously', function(done) {
    file.getStats('some/fake/path/to/fake/file.html', function (err, stats) {
      expect(err).not.to.be.null;
      expect(stats).to.be.undefined;
      done();
    });
  });

  it('should write a file', function () {
    var expected = testTxtContents;
    file.writeFileSync(testTxtWritePath, expected);
    var actual = file.readFileSync(testTxtWritePath);
    file.delete(testTxtWritePath);
    expect(actual).to.eql(expected);
  });

  it('should write a file (async)', function (done) {
    var expected = testTxtContents;
    file.writeFile(testTxtWritePath, expected, function () {
      file.readFile(testTxtWritePath, function (err, actual) {
        file.delete(testTxtWritePath);
        expect(actual).to.eql(expected);
        done();
      });
    });
  });

  it('should write the json file', function () {
    var expected = testJsonContents;
    file.writeJSONSync(testJsonWritePath, expected);
    var actual = file.readJSONSync(testJsonWritePath);
    file.delete(testJsonWritePath);
    expect(actual).to.eql(expected);
  });

  it('should write the json file (async)', function (done) {
    var expected = testJsonContents;
    file.writeJSON(testJsonWritePath, expected, function () {
      file.readJSON(testJsonWritePath, function (err, actual) {
        file.delete(testJsonWritePath);
        expect(actual).to.eql(expected);
        done();
      });
    });
  });

  it('should write the yaml file', function () {
    var expected = testYamlContents;
    file.writeYAMLSync(testYamlWritePath, expected);
    var actual = file.readYAMLSync(testYamlWritePath);
    file.delete(testYamlWritePath);
    expect(actual).to.eql(expected);
  });

  it('should write the yaml file (async)', function (done) {
    var expected = testYamlContents;
    file.writeYAML(testYamlWritePath, expected, function () {
      file.readYAML(testYamlWritePath, function (err, actual) {
        file.delete(testYamlWritePath);
        expect(actual).to.eql(expected);
        done();
      });
    });
  });

  it('should write the json file automatically', function () {
    var expected = testJsonContents;
    file.writeDataSync(testJsonWritePath, expected);
    var actual = file.readDataSync(testJsonWritePath);
    file.delete(testJsonWritePath);
    expect(actual).to.eql(expected);
  });

  it('should write the json file automatically (async)', function (done) {
    var expected = testJsonContents;
    file.writeData(testJsonWritePath, expected, function () {
      file.readData(testJsonWritePath, function (err, actual) {
        file.delete(testJsonWritePath);
        expect(actual).to.eql(expected);
        done();
      });
    });
  });

  it('should write the yaml file automatically', function () {
    var expected = testYamlContents;
    file.writeDataSync(testYamlWritePath, expected);
    var actual = file.readDataSync(testYamlWritePath);
    file.delete(testYamlWritePath);
    expect(actual).to.eql(expected);
  });

  it('should write the yaml file automatically (async)', function (done) {
    var expected = testYamlContents;
    file.writeData(testYamlWritePath, expected, function () {
      file.readData(testYamlWritePath, function (err, actual) {
        file.delete(testYamlWritePath);
        expect(actual).to.eql(expected);
        done();
      });
    });
  });

});