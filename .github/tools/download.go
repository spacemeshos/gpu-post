package main

import (
	"crypto/md5"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
)

func main() {
	fileUrl := os.Getenv("file-url")
	u, err := url.ParseRequestURI(fileUrl)
	if err != nil {
		log.Fatalf("failed to parse URL: %v\n", err)
	}

	fileName := os.Getenv("file-name")
	if fileName == "" {
		fileName = path.Base(u.Path)
	}

	location := os.Getenv("location")
	if location == "" {
		location, _ = os.Getwd()
	}

	if err := os.MkdirAll(location, 0o755); err != nil {
		log.Fatalf("failed to create directory at %s: %s", location, err)
	}

	expected_sha := os.Getenv("sha256")
	expected_md5 := os.Getenv("md5")

	out, err := os.Create(filepath.Join(location, fileName))
	if err != nil {
		log.Fatalf("failed to create file %s: %s", location, err)
	}
	defer out.Close()

	log.Println("Downloading file:")
	log.Println("\turl:", fileUrl)
	log.Println("\tname:", fileName)
	log.Println("\tlocation:", location)
	log.Println("\tMD5:", expected_md5)
	log.Println("\tSHA256:", expected_sha)

	resp, err := http.Get(fileUrl)
	if err != nil {
		log.Fatalf("failed to query url %s: %s\n", fileUrl, err)
	}
	defer resp.Body.Close()

	sha := sha256.New()
	md := md5.New()

	var r io.Reader = resp.Body
	r = io.TeeReader(r, sha)
	r = io.TeeReader(r, md)

	if _, err := io.Copy(out, r); err != nil {
		log.Fatalf("failed to write file: %s\n", err)
	}

	actual_sha := sha.Sum(nil)
	actual_md5 := md.Sum(nil)

	log.Println("Verifying file:")
	log.Println("\tFile:", out.Name())
	log.Printf("\tMD5: %x\n", actual_md5)
	log.Printf("\tSHA256: %x\n", actual_sha)

	status := 0
	if expected_sha != "" && expected_sha != hex.EncodeToString(actual_sha) {
		log.Println("\tactual SHA256 differs from expected SHA256")
		status = 1
	}

	if expected_md5 != "" && expected_md5 != hex.EncodeToString(actual_md5) {
		log.Println("\tactual MD5 differs from expected MD5")
		status = 1
	}
	os.Exit(status)
}
