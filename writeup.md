## Problem 1: Understanding Unicode
- ### What Unicode character does `chr(0)` return?
    -  the \x00 is returned - HEX 0.
        ```python
            >>> chr(0) # '\x00'
        ```
- ### How does this character's string representation differ from it's printed representation?
    - It is escaped and encapsulated in quote markings.
    ```python
        >>> chr(0).__repr__() # "'\\x00'"
    ```
- ### What happens when this character occurs in text?
    - When called in print statement it prints empty line 
    ```python
    >>> print(chr(0)) # Empty line
    ```
    - When concatenated with another string it's escaped hex form is within the string
    ```python
    >>> "this is a test" + chr(0) + "string" # 'this is a test\x00string'
    ```
    - When called in print statement and cocatenated with another string nothing happens
    ```python
    >>> print("this is a test" + chr(0) + "string") # this is a teststring
    ```

## Problem 2: Unicode Encodings
- ### What are some reasons to prefer training tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32?
    - The main reason would be that latter encodings require more bytes to represent the character resulting in longer sequences. Also the UTF-16 and UTF-32 when casted as list of bytes shows first two bytes are prefix of 255 and 254 - probably to mark the encoding. This can introduce some noise into the the encoded strings and decept the model during training.
- ### Consider the following incorrect function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results
```python 
def decode_utf8_bytes_to_str_wrond(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```
    
- This function is incorrect because it calls `decode()` method on each byte. This can result in error when string encoding requires more than one byte to represent string.
- Example string would be any string that contains character outside the range on base ASCII characters which can be represented by single byte, eg: `dzień dobry z łodzi`
```python
>>> decode_utf8_bytes_to_str_wrond("hello".encode("utf-8")) #'hello'
>>> decode_utf8_bytes_to_str_wrond("dzień dobry z łodzi".encode("utf-8")) #'UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc5 in position 0: unexpected end of data'
```