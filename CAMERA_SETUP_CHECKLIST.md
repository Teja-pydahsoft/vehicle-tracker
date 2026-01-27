# Camera Setup Checklist

Use this checklist to ensure all 4 cameras are properly configured and operational.

## Pre-Setup Requirements

### Hardware
- [ ] All 4 IP cameras are physically installed
- [ ] Cameras are powered on
- [ ] Network cables are connected
- [ ] Cameras are on the same network as the server
- [ ] Server meets minimum requirements:
  - [ ] CPU: Intel i5 or better
  - [ ] RAM: 8GB minimum (16GB recommended)
  - [ ] GPU: NVIDIA with CUDA (optional)
  - [ ] Network: Gigabit Ethernet

### Software
- [ ] Python 3.8+ installed
- [ ] Required packages installed (`pip install -r requirements.txt`)
- [ ] YOLO model downloaded (`yolov8n.pt`)
- [ ] Database initialized (`gate_log.db`)

## Camera Information Collection

### Camera 1 - Main Gate Entry
- [ ] Camera IP Address: ___________________
- [ ] Username: ___________________
- [ ] Password: ___________________
- [ ] RTSP Port: ___________________
- [ ] Stream Path: ___________________
- [ ] Full RTSP URL: rtsp://___:___@___:___/___
- [ ] Test with VLC/ffplay: [ ] Success [ ] Failed

### Camera 2 - Main Gate Exit
- [ ] Camera IP Address: ___________________
- [ ] Username: ___________________
- [ ] Password: ___________________
- [ ] RTSP Port: ___________________
- [ ] Stream Path: ___________________
- [ ] Full RTSP URL: rtsp://___:___@___:___/___
- [ ] Test with VLC/ffplay: [ ] Success [ ] Failed

### Camera 3 - Parking Entry
- [ ] Camera IP Address: ___________________
- [ ] Username: ___________________
- [ ] Password: ___________________
- [ ] RTSP Port: ___________________
- [ ] Stream Path: ___________________
- [ ] Full RTSP URL: rtsp://___:___@___:___/___
- [ ] Test with VLC/ffplay: [ ] Success [ ] Failed

### Camera 4 - Parking Exit
- [ ] Camera IP Address: ___________________
- [ ] Username: ___________________
- [ ] Password: ___________________
- [ ] RTSP Port: ___________________
- [ ] Stream Path: ___________________
- [ ] Full RTSP URL: rtsp://___:___@___:___/___
- [ ] Test with VLC/ffplay: [ ] Success [ ] Failed

## Network Configuration

- [ ] All cameras have static IP addresses
- [ ] Firewall allows RTSP traffic (port 554)
- [ ] Server can ping all camera IPs
- [ ] Network bandwidth is sufficient (minimum 10 Mbps per camera)

## Dashboard Configuration

### Initial Setup
- [ ] Launch API server: `python multi_camera_api.py`
- [ ] Access dashboard: http://localhost:5000
- [ ] Dashboard loads successfully

### Camera Configuration
- [ ] Navigate to Configuration tab
- [ ] Camera 1 configured:
  - [ ] Name entered
  - [ ] RTSP URL entered
  - [ ] Status set to Enabled
  - [ ] Test connection successful
- [ ] Camera 2 configured:
  - [ ] Name entered
  - [ ] RTSP URL entered
  - [ ] Status set to Enabled
  - [ ] Test connection successful
- [ ] Camera 3 configured:
  - [ ] Name entered
  - [ ] RTSP URL entered
  - [ ] Status set to Enabled
  - [ ] Test connection successful
- [ ] Camera 4 configured:
  - [ ] Name entered
  - [ ] RTSP URL entered
  - [ ] Status set to Enabled
  - [ ] Test connection successful
- [ ] Configuration saved successfully

### Global Settings
- [ ] Confidence threshold set: _____ %
- [ ] Processing rate set: Every _____ frame(s)
- [ ] Auto-restart enabled: [ ] Yes [ ] No
- [ ] OCR enabled: [ ] Yes [ ] No
- [ ] Settings saved

## Testing

### Individual Camera Tests
- [ ] Camera 1:
  - [ ] Started successfully
  - [ ] Video feed visible in Camera Grid
  - [ ] FPS showing (15-30 FPS)
  - [ ] Vehicles being detected
  - [ ] Direction (IN/OUT) working
  - [ ] Plate numbers detected (if OCR enabled)
- [ ] Camera 2:
  - [ ] Started successfully
  - [ ] Video feed visible in Camera Grid
  - [ ] FPS showing (15-30 FPS)
  - [ ] Vehicles being detected
  - [ ] Direction (IN/OUT) working
  - [ ] Plate numbers detected (if OCR enabled)
- [ ] Camera 3:
  - [ ] Started successfully
  - [ ] Video feed visible in Camera Grid
  - [ ] FPS showing (15-30 FPS)
  - [ ] Vehicles being detected
  - [ ] Direction (IN/OUT) working
  - [ ] Plate numbers detected (if OCR enabled)
- [ ] Camera 4:
  - [ ] Started successfully
  - [ ] Video feed visible in Camera Grid
  - [ ] FPS showing (15-30 FPS)
  - [ ] Vehicles being detected
  - [ ] Direction (IN/OUT) working
  - [ ] Plate numbers detected (if OCR enabled)

### System-Wide Tests
- [ ] All 4 cameras running simultaneously
- [ ] Dashboard statistics updating
- [ ] Total IN/OUT counts accurate
- [ ] Active cameras count correct (4/4)
- [ ] Average FPS reasonable (>15 FPS)
- [ ] Recent activity showing detections
- [ ] Database logging working
- [ ] History tab showing records
- [ ] Filters working correctly
- [ ] Export functions working

## Performance Optimization

### If FPS is Low (<15 FPS)
- [ ] Increase processing rate (every 2nd or 3rd frame)
- [ ] Reduce camera resolution
- [ ] Disable OCR temporarily
- [ ] Check CPU/GPU usage
- [ ] Verify network bandwidth

### If Detection Accuracy is Low
- [ ] Adjust confidence threshold
- [ ] Check camera angles
- [ ] Improve lighting conditions
- [ ] Clean camera lenses
- [ ] Adjust detection zones

### If Plate Recognition is Poor
- [ ] Ensure cameras are close enough to vehicles
- [ ] Check image quality/resolution
- [ ] Adjust OCR confidence threshold
- [ ] Verify lighting (especially at night)
- [ ] Consider adding IR illuminators

## Maintenance Schedule

### Daily
- [ ] Check all cameras are active
- [ ] Verify FPS is normal
- [ ] Review detection accuracy
- [ ] Check for any errors in logs

### Weekly
- [ ] Clean camera lenses
- [ ] Review and export logs
- [ ] Check database size
- [ ] Verify backup (if enabled)

### Monthly
- [ ] Update camera firmware
- [ ] Review and optimize settings
- [ ] Clean up old logs (if not auto-cleanup)
- [ ] Test failover/restart procedures

## Troubleshooting Reference

### Camera Won't Start
1. Check RTSP URL format
2. Verify camera is online (ping IP)
3. Test with VLC/ffplay
4. Check credentials
5. Review firewall settings

### Low FPS
1. Increase processing rate
2. Reduce resolution
3. Check network bandwidth
4. Monitor CPU/GPU usage
5. Restart camera

### No Detections
1. Check camera view/angle
2. Verify detection zone
3. Lower confidence threshold
4. Check lighting conditions
5. Review YOLO model

### Database Errors
1. Check disk space
2. Verify database permissions
3. Backup and recreate database
4. Check for corruption

## Support Contacts

**System Administrator:** ___________________
**Network Administrator:** ___________________
**Camera Vendor Support:** ___________________
**Emergency Contact:** ___________________

## Notes

_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________

---

**Checklist Version:** 1.0  
**Last Updated:** January 25, 2026  
**Completed By:** ___________________  
**Date:** ___________________  
**Signature:** ___________________
