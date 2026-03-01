"""
Email Notification System
Handles sending emails for orders, payments, and account activities
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from typing import Optional, Dict, List
import threading
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class EmailService:
    """Email service using SMTP"""
    
    def __init__(self, 
                 smtp_host: str,
                 smtp_port: int,
                 smtp_user: str,
                 smtp_password: str,
                 from_email: str,
                 from_name: str = "Tomato AI"):
        """
        Initialize email service
        
        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            from_email: Sender email address
            from_name: Sender display name
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.from_name = from_name
        self.enabled = bool(smtp_host and smtp_user and smtp_password)
        
        if self.enabled:
            logger.info(f"Email service initialized: {from_email}")
        else:
            logger.warning("Email service disabled - missing SMTP configuration")
    
    def send_email(self,
                   to_email: str,
                   subject: str,
                   html_body: str,
                   text_body: Optional[str] = None,
                   attachments: Optional[List[Dict]] = None,
                   async_send: bool = True) -> bool:
        """
        Send an email
        
        Args:
            to_email: Recipient email
            subject: Email subject
            html_body: HTML email body
            text_body: Plain text fallback (optional)
            attachments: List of attachments (optional)
            async_send: Send in background thread
            
        Returns:
            True if sent successfully (or queued)
        """
        if not self.enabled:
            logger.warning(f"Email not sent (disabled): {subject} to {to_email}")
            return False
        
        if async_send:
            # Send in background thread
            thread = threading.Thread(
                target=self._send_email_sync,
                args=(to_email, subject, html_body, text_body, attachments)
            )
            thread.daemon = True
            thread.start()
            logger.info(f"Email queued: {subject} to {to_email}")
            return True
        else:
            return self._send_email_sync(to_email, subject, html_body, text_body, attachments)
    
    def _send_email_sync(self,
                        to_email: str,
                        subject: str,
                        html_body: str,
                        text_body: Optional[str] = None,
                        attachments: Optional[List[Dict]] = None) -> bool:
        """
        Send email synchronously (internal method)
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            msg['Date'] = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S +0000')
            
            # Add text part (fallback)
            if text_body:
                text_part = MIMEText(text_body, 'plain', 'utf-8')
                msg.attach(text_part)
            
            # Add HTML part
            html_part = MIMEText(html_body, 'html', 'utf-8')
            msg.attach(html_part)
            
            # Add attachments if any
            if attachments:
                for attachment in attachments:
                    # TODO: Implement attachment handling
                    pass
            
            # Connect and send
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.set_debuglevel(0)
                server.ehlo()
                
                # Use TLS if port 587
                if self.smtp_port == 587:
                    server.starttls()
                    server.ehlo()
                
                # Login
                server.login(self.smtp_user, self.smtp_password)
                
                # Send
                server.send_message(msg)
            
            logger.info(f"Email sent successfully: {subject} to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
            return False


class EmailTemplates:
    """Email HTML templates"""
    
    @staticmethod
    def get_base_template() -> str:
        """Base email template with styling"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 600px;
            margin: 20px auto;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 28px;
        }
        .content {
            padding: 30px;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }
        .button {
            display: inline-block;
            padding: 12px 30px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: 600;
            margin: 20px 0;
        }
        .info-box {
            background: #f8f9fa;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .order-details {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .order-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .order-row:last-child {
            border-bottom: none;
            font-weight: bold;
            font-size: 1.2em;
        }
        .icon {
            font-size: 48px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""
    
    @staticmethod
    def order_confirmation(order_data: Dict) -> tuple:
        """
        Generate order confirmation email
        
        Args:
            order_data: Order information dict
            
        Returns:
            (subject, html_body, text_body)
        """
        order_id = order_data.get('id', 'N/A')
        customer_name = order_data.get('customer_name', 'Khách hàng')
        total = order_data.get('total', 0)
        items = order_data.get('items', [])
        payment_method = order_data.get('payment_method', 'cod')
        
        # Payment method display
        payment_display = {
            'cod': 'Thanh toán khi nhận hàng (COD)',
            'vnpay': 'VNPay',
            'momo': 'MoMo'
        }.get(payment_method, payment_method)
        
        # Build items HTML
        items_html = ""
        for item in items:
            item_total = item.get('price', 0) * item.get('quantity', 0)
            items_html += f"""
            <div class="order-row">
                <span>{item.get('name', 'Sản phẩm')} x {item.get('quantity', 0)}</span>
                <span>{item_total:,.0f} đ</span>
            </div>
            """
        
        content = f"""
        <div class="header">
            <div class="icon">📦</div>
            <h1>Xác nhận đơn hàng</h1>
        </div>
        <div class="content">
            <p>Xin chào <strong>{customer_name}</strong>,</p>
            
            <p>Cảm ơn bạn đã đặt hàng tại <strong>Tomato AI</strong>! Đơn hàng của bạn đã được tiếp nhận và đang được xử lý.</p>
            
            <div class="info-box">
                <strong>📋 Mã đơn hàng:</strong> #{order_id}<br>
                <strong>💳 Phương thức thanh toán:</strong> {payment_display}<br>
                <strong>📅 Thời gian:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}
            </div>
            
            <h3>Chi tiết đơn hàng:</h3>
            <div class="order-details">
                {items_html}
                <div class="order-row">
                    <span><strong>Tổng cộng:</strong></span>
                    <span><strong>{total:,.0f} đ</strong></span>
                </div>
            </div>
            
            <p>Chúng tôi sẽ liên hệ với bạn sớm nhất để xác nhận và giao hàng.</p>
            
            <center>
                <a href="http://localhost:5000/profile" class="button">Xem đơn hàng</a>
            </center>
        </div>
        <div class="footer">
            <p><strong>Tomato AI - Hệ thống phát hiện bệnh cà chua thông minh</strong></p>
            <p>📧 support@tomatoai.com | 📞 1900-xxxx-xx</p>
            <p>Email này được gửi tự động, vui lòng không trả lời.</p>
        </div>
        """
        
        html = EmailTemplates.get_base_template().replace('{content}', content)
        text = f"Đơn hàng #{order_id} đã được xác nhận. Tổng tiền: {total:,.0f} đ"
        subject = f"Xác nhận đơn hàng #{order_id} - Tomato AI"
        
        return subject, html, text
    
    @staticmethod
    def payment_success(payment_data: Dict) -> tuple:
        """
        Generate payment success email
        
        Args:
            payment_data: Payment information dict
            
        Returns:
            (subject, html_body, text_body)
        """
        order_id = payment_data.get('order_id', 'N/A')
        amount = payment_data.get('amount', 0)
        payment_method = payment_data.get('payment_method', 'online')
        transaction_id = payment_data.get('transaction_id', 'N/A')
        
        content = f"""
        <div class="header">
            <div class="icon">✅</div>
            <h1>Thanh toán thành công</h1>
        </div>
        <div class="content">
            <p>Xin chào,</p>
            
            <p>Giao dịch thanh toán của bạn đã được xử lý <strong>thành công</strong>!</p>
            
            <div class="info-box">
                <strong>📦 Mã đơn hàng:</strong> #{order_id}<br>
                <strong>💰 Số tiền:</strong> {amount:,.0f} đ<br>
                <strong>💳 Phương thức:</strong> {payment_method}<br>
                <strong>🔖 Mã giao dịch:</strong> {transaction_id}<br>
                <strong>📅 Thời gian:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            </div>
            
            <p>Đơn hàng của bạn sẽ được chuẩn bị và giao sớm nhất có thể.</p>
            
            <center>
                <a href="http://localhost:5000/profile" class="button">Xem chi tiết</a>
            </center>
            
            <p style="margin-top: 30px; font-size: 0.9em; color: #666;">
                <strong>Lưu ý:</strong> Vui lòng giữ email này làm chứng từ thanh toán.
            </p>
        </div>
        <div class="footer">
            <p><strong>Tomato AI - Hệ thống phát hiện bệnh cà chua thông minh</strong></p>
            <p>📧 support@tomatoai.com | 📞 1900-xxxx-xx</p>
        </div>
        """
        
        html = EmailTemplates.get_base_template().replace('{content}', content)
        text = f"Thanh toán thành công cho đơn hàng #{order_id}. Số tiền: {amount:,.0f} đ"
        subject = f"✅ Thanh toán thành công - Đơn hàng #{order_id}"
        
        return subject, html, text
    
    @staticmethod
    def payment_failed(order_id: str, reason: str = "Không xác định") -> tuple:
        """
        Generate payment failed email
        
        Returns:
            (subject, html_body, text_body)
        """
        content = f"""
        <div class="header" style="background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);">
            <div class="icon">❌</div>
            <h1>Thanh toán thất bại</h1>
        </div>
        <div class="content">
            <p>Xin chào,</p>
            
            <p>Rất tiếc, giao dịch thanh toán của bạn <strong>không thành công</strong>.</p>
            
            <div class="info-box" style="border-left-color: #f44336;">
                <strong>📦 Mã đơn hàng:</strong> #{order_id}<br>
                <strong>⚠️ Lý do:</strong> {reason}<br>
                <strong>📅 Thời gian:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            </div>
            
            <p>Bạn có thể thử lại với phương thức thanh toán khác hoặc liên hệ với chúng tôi để được hỗ trợ.</p>
            
            <center>
                <a href="http://localhost:5000/cart" class="button" style="background: #f44336;">Thử lại</a>
            </center>
        </div>
        <div class="footer">
            <p><strong>Tomato AI</strong></p>
            <p>📧 support@tomatoai.com | 📞 1900-xxxx-xx</p>
        </div>
        """
        
        html = EmailTemplates.get_base_template().replace('{content}', content)
        text = f"Thanh toán thất bại cho đơn hàng #{order_id}. Lý do: {reason}"
        subject = f"❌ Thanh toán thất bại - Đơn hàng #{order_id}"
        
        return subject, html, text
    
    @staticmethod
    def welcome_email(user_name: str, user_email: str) -> tuple:
        """Generate welcome email for new users"""
        content = f"""
        <div class="header">
            <div class="icon">👋</div>
            <h1>Chào mừng đến với Tomato AI</h1>
        </div>
        <div class="content">
            <p>Xin chào <strong>{user_name}</strong>,</p>
            
            <p>Cảm ơn bạn đã đăng ký tài khoản tại <strong>Tomato AI</strong>!</p>
            
            <p>Với tài khoản của mình, bạn có thể:</p>
            <ul>
                <li>🔍 Phát hiện bệnh trên lá cà chua bằng AI</li>
                <li>💬 Chat với AI về cách chăm sóc cà chua</li>
                <li>🎮 Chơi game và nhận voucher</li>
                <li>🛒 Mua sắm thuốc trừ sâu, phân bón</li>
                <li>📊 Theo dõi lịch sử dự đoán và đơn hàng</li>
            </ul>
            
            <center>
                <a href="http://localhost:5000/" class="button">Bắt đầu sử dụng</a>
            </center>
            
            <p style="margin-top: 30px; color: #666;">
                Nếu bạn cần hỗ trợ, đừng ngại liên hệ với chúng tôi!
            </p>
        </div>
        <div class="footer">
            <p><strong>Tomato AI - Hệ thống phát hiện bệnh cà chua thông minh</strong></p>
            <p>📧 support@tomatoai.com | 📞 1900-xxxx-xx</p>
        </div>
        """
        
        html = EmailTemplates.get_base_template().replace('{content}', content)
        text = f"Chào mừng {user_name} đến với Tomato AI!"
        subject = "👋 Chào mừng đến với Tomato AI"
        
        return subject, html, text


# Singleton email service instance
_email_service: Optional[EmailService] = None


def init_email_service(smtp_host: str, smtp_port: int, smtp_user: str, 
                       smtp_password: str, from_email: str, from_name: str = "Tomato AI"):
    """Initialize global email service"""
    global _email_service
    _email_service = EmailService(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        from_email=from_email,
        from_name=from_name
    )
    return _email_service


def get_email_service() -> Optional[EmailService]:
    """Get global email service instance"""
    return _email_service


def send_order_confirmation_email(to_email: str, order_data: Dict) -> bool:
    """Send order confirmation email"""
    service = get_email_service()
    if not service:
        return False
    
    subject, html, text = EmailTemplates.order_confirmation(order_data)
    return service.send_email(to_email, subject, html, text)


def send_payment_success_email(to_email: str, payment_data: Dict) -> bool:
    """Send payment success email"""
    service = get_email_service()
    if not service:
        return False
    
    subject, html, text = EmailTemplates.payment_success(payment_data)
    return service.send_email(to_email, subject, html, text)


def send_payment_failed_email(to_email: str, order_id: str, reason: str) -> bool:
    """Send payment failed email"""
    service = get_email_service()
    if not service:
        return False
    
    subject, html, text = EmailTemplates.payment_failed(order_id, reason)
    return service.send_email(to_email, subject, html, text)


def send_welcome_email(to_email: str, user_name: str) -> bool:
    """Send welcome email to new user"""
    service = get_email_service()
    if not service:
        return False
    
    subject, html, text = EmailTemplates.welcome_email(user_name, to_email)
    return service.send_email(to_email, subject, html, text)
