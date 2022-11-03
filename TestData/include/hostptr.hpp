#pragma once

#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
using namespace std;

template<typename T>
class HostPtr
{
public:
    HostPtr():m_Size(0), m_Bytes(0) {}

    HostPtr(size_t size, bool bZeroInit=true)
    {
        m_Size = size;
        m_Bytes = size * sizeof(T);

        m_Ptr = new T[m_Size];
        if (bZeroInit)
            memset(m_Ptr, 0, m_Bytes);
    }

    HostPtr(const T* p, size_t size)
    {
        m_Size = size;
        m_Bytes = size * sizeof(T);

        m_Ptr = new T[m_Size];
        memcpy(m_Ptr, p, m_Bytes);
    }

    void Resize(size_t size)
    {
        if (m_Size == size)
            return;

        if (m_Size>0)
            delete []m_Ptr;
        
        m_Size = size;
        m_Bytes = size * sizeof(T);

        m_Ptr = new T[m_Size];
    }

    T* GetPtr()
    {
        return m_Ptr;
    }

    void SetZeros()
    {
        if (m_Size>0)
            memset(m_Ptr, 0, m_Bytes);
        else{
            std::cerr<<"ERROR! HostPtr::SetZeros() called by uninitialized pointer!!!"<<std::endl;
        }
    }

    const T* GetPtrConst() const
    {
        return (const T*)m_Ptr;
    }

    T& operator ()(size_t idx)
    {
        return m_Ptr[idx];
    }

    const T& operator ()(size_t idx) const
    {
        return m_Ptr[idx];
    }

    size_t GetSize() const
    {
        return m_Size;
    }

    void LoadFromFile(const char* fname, size_t length=0)
    {
        std::ifstream ifs;
        ifs.open (fname, std::ifstream::binary);
        if (!ifs)
        {
            std::cout<<"ERROR! Cannot open input file : "<<fname<<std::endl;
            return;
        }

        std::cout<<"Loading from file "<<fname<<" ..."<<std::endl;

        size_t len = length;
        if (len == 0)
            len = m_Bytes;

        ifs.read((char*)m_Ptr, len);
        ifs.close();
    }

    void SaveToFile(const char* fname) const
    {
        std::ofstream ofs;
        ofs.open (fname, std::ofstream::binary);
        if (!ofs)
        {
            std::cout<<"ERROR! Cannot open output file : "<<fname<<std::endl;
            return;
        }

        std::cout<<"Saving to file "<<fname<<" ..."<<std::endl;

        ofs.write((const char*)m_Ptr, m_Bytes);
        ofs.close();
    }

    void DumpVal(size_t len=0, int disp_width=12, int elem_per_line=8)
    {
        if(len==0)
            len = m_Size;

        for(size_t i=0; i<len; i++)
        {
            // disp index in front of every line
            if(i%elem_per_line==0)
                std::cout<<std::setw(8)<<i<<" : ";

            std::cout<<std::setw(disp_width)<<m_Ptr[i]<<"    ";

            if ((i+1)%elem_per_line==0)
                std::cout<<std::endl;
        }
    }

    void DumpFormatted(size_t len=0, int elem_per_line=8, const char* format="%#8x ")
    {
        if(len==0)
            len = m_Size;

        for(size_t i=0; i<len; i++)
        {
            // disp index in front of every line
            if(i%elem_per_line==0)
                printf("%4zu : ", i);

            printf(format, m_Ptr[i]);

            if ((i+1)%elem_per_line==0)
                printf("\n");
        }
    }

    ~HostPtr<T>()
    {
        if (m_Size > 0)
        {
            delete []m_Ptr;
            m_Ptr = 0;
        }
    }

private:
    T* m_Ptr;        // host pointer
    size_t m_Size;   // number of elements, not bytes
    size_t m_Bytes;  // number of bytes
};

