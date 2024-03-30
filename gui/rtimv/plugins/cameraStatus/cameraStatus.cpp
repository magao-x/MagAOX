
#include "cameraStatus.hpp"

#define errPrint(expl) std::cerr << "cameraStatus: " << __FILE__ << " " << __LINE__ << " " << expl << std::endl;

cameraStatus::cameraStatus() : rtimvOverlayInterface()
{
}

cameraStatus::~cameraStatus()
{
}

int cameraStatus::attachOverlay( rtimvOverlayAccess & roa,
                                 mx::app::appConfigurator & config
                               )
{   
    m_roa = roa;
    m_qgs = roa.m_graphicsView->scene();
    
    
    config.configUnused(m_deviceName, mx::app::iniFile::makeKey("camera", "name"));
    
    if(m_deviceName == "")
    {
       m_enableable = false;
       disableOverlay();
       return 1; //Tell rtimv to unload me since not configured.
    }
    
    m_enableable = true;
    m_enabled = true;
    
    config.configUnused(m_filterDeviceNames, mx::app::iniFile::makeKey("camera", "filterDevices"));

    if(m_roa.m_dictionary != nullptr)
    {
        //Register these
        (*m_roa.m_dictionary)[m_deviceName + ".temp_ccd.current"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".fg_frameSize.width"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".fg_frameSize.height"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_w.current"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_h.current"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_x.current"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_y.current"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_bin_y.current"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_bin_x.current"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_w.target"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_h.target"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_x.target"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_y.target"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_bin_y.target"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".roi_region_bin_x.target"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".exptime.current"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".fps.current"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".emgain.current"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".shutter_status.status"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".shutter.toggle"].setBlob(nullptr, 0);
      
        m_presetNames.resize(m_filterDeviceNames.size());

        for(size_t f = 0; f < m_filterDeviceNames.size(); ++f)
        {
            if(m_filterDeviceNames[f].find("fw") == 0) m_presetNames[f] = ".filterName";
            else if(m_filterDeviceNames[f].find("flip") == 0) m_presetNames[f] = ".presetName";
            else m_presetNames[f] = ".presetName";

            (*m_roa.m_dictionary)[m_filterDeviceNames[f] + ".fsm.state"].setBlob(nullptr, 0);
            (*m_roa.m_dictionary)[m_filterDeviceNames[f] + m_presetNames[f]].setBlob(nullptr, 0);
        }


        (*m_roa.m_dictionary)[m_deviceName + "-sw.writing.toggle"].setBlob(nullptr, 0);

      //(*m_roa.m_dictionary)[m_deviceName + ""].setBlob(nullptr, 0);

   }

   connect(this, SIGNAL(newStretchBox(StretchBox *)), m_roa.m_mainWindowObject, SLOT(addStretchBox(StretchBox *)));
   connect(this, SIGNAL(savingState(bool)), m_roa.m_mainWindowObject, SLOT(savingState(bool)));

   if(m_enabled) enableOverlay();
   else disableOverlay();
   
   return 0;
}

bool cameraStatus::blobExists( const std::string & propel )
{
    if(m_roa.m_dictionary->count(m_deviceName + "." + propel) == 0)
    {
        return false;
    }

    if((*m_roa.m_dictionary)[m_deviceName + "." + propel].getBlobSize() == 0)
    {
        return false;
    }

    return true;
}

bool cameraStatus::getBlobStr( const std::string & deviceName, const std::string & propel )
{
    if(m_roa.m_dictionary->count(deviceName + "." + propel) == 0)
    {
        return false;
    }

    if( ((*m_roa.m_dictionary)[deviceName + "." + propel].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) )
    {
        return false;
    }

    if(m_blob[0] == '\0')
    {
        return false;
    }

    return true;
}

bool cameraStatus::getBlobStr( const std::string & propel )
{
    return getBlobStr(m_deviceName, propel);
}

template<>
int cameraStatus::getBlobVal<int>( const std::string & propel, int defVal )
{
    if(getBlobStr(propel))
    {
        return atoi(m_blob);
    }
    else
    {
        return defVal;
    }
}

template<>
float cameraStatus::getBlobVal<float>( const std::string & propel, float defVal )
{
    if(getBlobStr(propel))
    {
        return strtod(m_blob,0);
    }
    else
    {
        return defVal;
    }
}


int cameraStatus::updateOverlay()
{
    if(!m_enabled) return 0;
   
    if(m_roa.m_dictionary == nullptr) return 0;
   
    if(m_roa.m_graphicsView == nullptr) return 0;

    size_t n = 0;
    //char * str;
    char tstr[128];
    std::string sstr;
   
    if(getBlobStr("temp_ccd.current"))
    {
        snprintf(tstr, sizeof(tstr), "%0.1f C", strtod(m_blob,0));
        m_roa.m_graphicsView->statusTextText(n, tstr);
        ++n;
    }
    if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;
   
    //Get curr size
    m_width = -1;
    if(getBlobStr("fg_frameSize.width"))
    {
        m_width = atoi(m_blob);
    }

    m_height=-1;
    if(getBlobStr("fg_frameSize.height"))
    {
        m_height = atoi(m_blob);
    }
   
    if(blobExists("roi_region_w.current") && blobExists("roi_region_h.current") && blobExists("roi_region_x.current") 
                          && blobExists("roi_region_y.current") && blobExists("roi_region_bin_x.current") && blobExists("roi_region_bin_y.current"))
    {
        int w = getBlobVal<int>("roi_region_w.current", -1);
        int h = getBlobVal<int>("roi_region_h.current", -1);
        int ibx = getBlobVal<int>("roi_region_bin_x.current", -1);
        int iby = getBlobVal<int>("roi_region_bin_y.current", -1);
        float x = getBlobVal<float>("roi_region_x.current", -1);
        float y = getBlobVal<float>("roi_region_y.current", -1);

        //Only go on if we got good values
        if(w > 0 && h > 0 && x > 0 && y > 0 && ibx > 0 && iby > 0)
        {
            snprintf(tstr, sizeof(tstr), "%dx%d [%dx%d]", w, h, ibx, iby );

            //**************** 
            m_roa.m_graphicsView->statusTextText(n, tstr);
            ++n;
            
            float bx = ibx;
            float by = iby;

            if(blobExists("roi_region_w.target") && blobExists("roi_region_h.target") && blobExists("roi_region_x.target") 
                          && blobExists("roi_region_y.target") && blobExists("roi_region_bin_x.target") && blobExists("roi_region_bin_y.target"))
            {

                std::lock_guard<std::mutex> guard(m_roiBoxMutex);

                int wt = getBlobVal<int>("roi_region_w.target", -1);
                int ht = getBlobVal<int>("roi_region_h.target", -1);
                float bxt = getBlobVal<float>("roi_region_bin_x.target", -1);
                float byt = getBlobVal<float>("roi_region_bin_y.target", -1);
                float xt = getBlobVal<float>("roi_region_x.target", -1);
                float yt = getBlobVal<float>("roi_region_y.target", -1);

                if( (wt > 0 && ht > 0 && xt > 0 && yt > 0 && bxt >= 1 && byt >= 1) && ( wt != w || ht != h || xt != x || yt != y))
                {
                    if(!m_roiBox)
                    {
                        m_roiBox = new StretchBox(xt-0.5*(wt-1),yt-0.5*(ht-1),wt,ht);
                        m_roiBox->setPenColor(Qt::magenta);
                        m_roiBox->setPenWidth(1);
                        m_roiBox->setVisible(true);
                        m_roiBox->setStretchable(false);
                        m_roiBox->setRemovable(false);
                        std::cerr << "Connecting\n";
                        connect(m_roiBox, SIGNAL(remove(StretchBox*)), this, SLOT(stretchBoxRemove(StretchBox*)));
                        emit newStretchBox(m_roiBox);
                    }
                    //note: can only be here if bx > 0
                    float xc = (xt*bxt-x*bx + 0.5*((float)m_width*bx-1))/bx;
                    float yc = (m_height*by - ( yt*byt-y*by + 0.5*((float)m_height*by-1)))/by;
            
                    float bxleftX = xc-0.5*(wt*bxt/bx-1.0);
                    float bxleftY = yc-0.5*(ht*byt/by-1.0);
                    float bxW = wt*bxt/bx;
                    float bxH = ht*byt/by;

                    m_roiBox->setRect(m_roiBox->mapRectFromScene(bxleftX,bxleftY,bxW,bxH));
                    m_roiBox->setVisible(true);
                }
                else
                {
                    if(m_roiBox) m_roiBox->setVisible(false);
                }

            }//if(blobExists("roi_region_w.target")
      
        }//if(w > 0 && h > 0 && x > 0 && y > 0 && ibx > 0 && iby > 0)

    }//if(blobExists("roi_region_w.current") ...

    //***********************
    if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;
   

    float et = getBlobVal<float>("exptime.current", -1);
    if(et >= 0)
    {
        if(et >= 100)
        {
            snprintf(tstr, sizeof(tstr), "%0.1f s", et);
        }
        else if(et >= 10)
        {
            snprintf(tstr, sizeof(tstr), "%0.2f s", et);
        }
        else if(et >= 0.1)
        {
            snprintf(tstr, sizeof(tstr), "%0.3f s", et);
        }
        else
        {
            if(et >= 0.00009999)
            {
                snprintf(tstr, sizeof(tstr), "%0.2f ms", et*1000.0);
            }
            else
            {
                snprintf(tstr, sizeof(tstr), "%0.2f us", et*1000000.0);
            }
        } 
        
        m_roa.m_graphicsView->statusTextText(n, tstr);
        ++n;
    }
    if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;
   
    float fps = getBlobVal<float>("fps.current",-1);
    if(fps >= 0)
    {
        snprintf(tstr, sizeof(tstr), "%0.1f FPS", fps);
        m_roa.m_graphicsView->statusTextText(n, tstr);
        ++n;
    }
    if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;

    int emg = getBlobVal<int>("emgain.current", -1);
    if(emg >= 0)
    {
        snprintf(tstr, sizeof(tstr), "EMG: %d", emg);
        m_roa.m_graphicsView->statusTextText(n, tstr);
        ++n;
    }
    if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;
   
    for(size_t f = 0; f < m_filterDeviceNames.size(); ++f)
    {
        if(getBlobStr(m_filterDeviceNames[f], "fsm.state"))
        {
            std::string fwstate = std::string(m_blob);
        
            if(fwstate == "READY" || fwstate == "OPERATING")
            {
                dictionaryIteratorT start = m_roa.m_dictionary->lower_bound(m_filterDeviceNames[f] + m_presetNames[f]);
                dictionaryIteratorT end = m_roa.m_dictionary->upper_bound(m_filterDeviceNames[f] + m_presetNames[f] + "Z");
               
                std::string fkey;
                while(start != end)
                {
                    if( (start->second.getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) errPrint("bad string"); //Don't trust this as a string
                  
                    if(m_blob[0] != '\0')
                    {
                        if(std::string(m_blob) == "on")
                        {
                            fkey = start->first;
                            break;
                        }
                    }
                    ++start;
                }

                size_t pp = fkey.rfind('.');
                std::string filn;
                if(pp != std::string::npos && pp < fkey.size()-1) //need to be able to add 1
                {
                    filn = m_filterDeviceNames[f] + ": " + fkey.substr(pp+1).c_str();
                }
                else
                {
                    filn = m_filterDeviceNames[f] + ": unk";
                }
                m_roa.m_graphicsView->statusTextText(n, filn.c_str());
                ++n;
            }
            else
            {
                fwstate = m_filterDeviceNames[f] +  ": " + fwstate;
                m_roa.m_graphicsView->statusTextText(n, fwstate.c_str());
                ++n;
            }
        }
        if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;   
    }

    if(getBlobStr("shutter_status.status"))
    {
        sstr = std::string(m_blob);
      
        if(sstr == "READY")
        {
            if(getBlobStr("shutter.toggle"))
            {
                sstr = std::string(m_blob);
                if(sstr == "on") sstr = "sh SHUT";
                else sstr = "sh OPEN";
                m_roa.m_graphicsView->statusTextText(n, sstr.c_str());
                ++n;
            }
        }
        else
        {

            sstr = "sh " + sstr;
            m_roa.m_graphicsView->statusTextText(n, sstr.c_str());
            ++n;
        }
    }    
    if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;  

    if(getBlobStr(m_deviceName + "-sw", "writing.toggle"))
    {
        sstr = std::string(m_blob);
        if(sstr == "on") emit savingState(true);
        else emit savingState(false);
    }

    return 0;
}

void cameraStatus::keyPressEvent( QKeyEvent * ke)
{
   char key = ke->text()[0].toLatin1();
   
   if(key == 'C')
   {
      if(m_enabled) disableOverlay();
      else enableOverlay();
   }
   else if(key == 'R')
   {
      std::cerr << "ROI\n";
   }
}

bool cameraStatus::overlayEnabled()
{
   return m_enabled;
}

void cameraStatus::enableOverlay()
{
   if(m_enableable == false) return;
   
   m_enabled = true;
}

void cameraStatus::disableOverlay()
{
   for(size_t n=0; n<m_roa.m_graphicsView->statusTextNo();++n)
   {
      m_roa.m_graphicsView->statusTextText(n, "");
   }
   
   m_enabled = false;
}

void cameraStatus::stretchBoxRemove(StretchBox * sb)
{
    std::cerr << "cameraStatus::stretchBoxRemove 1\n";
    std::lock_guard<std::mutex> guard(m_roiBoxMutex);
    if(!m_roiBox)
    {
        return;
    }

    std::cerr << "cameraStatus::stretchBoxRemove 2\n";

    if(sb != m_roiBox)
    {
        return;
    }

    std::cerr << "cameraStatus::stretchBoxRemove 3\n";

    m_roiBox = nullptr;
}

std::vector<std::string> cameraStatus::info()
{
    std::vector<std::string> vinfo;
    vinfo.push_back("Camera Status overlay: " + m_deviceName);

    return vinfo;
}